// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ParserResult, ReasoningParser};

use minijinja::{Environment, context};

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

fn write_to_stdin(mut stdin: ChildStdin, rx: Receiver<String>) {
    while let Ok(line) = rx.recv() {
        stdin
            .write_all(line.as_bytes())
            .expect("Failed to write to stdin");
        stdin.flush().expect("Failed to flush stdin");
    }
}

fn read_from_stdout(stdout: ChildStdout, tx: Sender<String>) {
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        match line {
            Ok(line) => {
                tx.send(line).expect("Failed to send stdout line");
            }
            Err(_) => break,
        }
    }
}

// define a jinja template in static string
pub const REASONING_PYTHON_TEMPLATE: &str = r#"
import os
import sys
# get absolute path
PATH = "{{ path }}"
PATH = os.path.abspath(PATH)


#extract parent directory
parent_dir = os.path.dirname(PATH)

#extract filename without extension
module_name, _ = os.path.splitext(os.path.basename(PATH))

sys.path.append(parent_dir)

#import the module
reasoning_module = __import__(module_name)

# find class with BaseReasoningParser in its bases

cls_name = ""

import inspect

for name, obj in vars(reasoning_module).items():
    if inspect.isclass(obj) and hasattr(obj, "__bases__"):
        # Check if BaseReasoningParser is anywhere in the MRO
        if any("BaseReasoningParser" in base.__name__ for base in inspect.getmro(obj)[1:]):
            cls_name = name
            break

if cls_name == "":
    raise ValueError("No class with BaseReasoningParser in its bases found in the module")

ReasoningParserClass = getattr(reasoning_module, cls_name)

parser_instance = ReasoningParserClass()

import json
text_input = json.loads(r'''{{ text|tojson }}''')
token_ids_input = {{ token_ids|tojson }}

normal_text, reasoning_text = parser_instance.detect_and_parse_reasoning(text_input, token_ids_input)

print(f"{normal_text}")
print(f"{reasoning_text}")
"#;

pub const REASONING_PYTHON_TEMPLATE_STREAMING: &str = r#"
import os
import sys
# get absolute path
PATH = "{{ path }}"
PATH = os.path.abspath(PATH)


#extract parent directory
parent_dir = os.path.dirname(PATH)

#extract filename without extension
module_name, _ = os.path.splitext(os.path.basename(PATH))

sys.path.append(parent_dir)

#import the module
reasoning_module = __import__(module_name)

# find class with BaseReasoningParser in its bases

cls_name = ""

import inspect

for name, obj in vars(reasoning_module).items():
    if inspect.isclass(obj) and hasattr(obj, "__bases__"):
        # Check if BaseReasoningParser is anywhere in the MRO
        if any("BaseReasoningParser" in base.__name__ for base in inspect.getmro(obj)[1:]):
            cls_name = name
            break

if cls_name == "":
    raise ValueError("No class with BaseReasoningParser in its bases found in the module")

ReasoningParserClass = getattr(reasoning_module, cls_name)

parser_instance = ReasoningParserClass()

# read text and token_ids from sys.stdin in a loop
for line in sys.stdin:
    if not line:
        break

    text, token_ids = line.rsplit("],[", 1)

    token_ids = list(map(int, token_ids.strip().split(","))) if token_ids.strip() else []
    normal_text, reasoning_text = parser_instance.parse_reasoning_streaming_incremental(text, token_ids)

    print(f"{normal_text}")
    print(f"{reasoning_text}")
    sys.stdout.flush()
"#;

#[derive(std::fmt::Debug)]
pub struct PythonProcessParser {
    path: String,
    child: std::process::Child,
    tx_in: Sender<String>,
    rx_out: Receiver<String>,
}

impl PythonProcessParser {
    pub fn new(path: &str) -> Self {
        let script = PythonProcessParser::render_script_streaming(path);
        let mut child = Command::new("python3")
            .arg("-u") // unbuffered output so we can read stdout line-by-line
            .arg("-c")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        let stdin = child.stdin.take().expect("Failed to open stdin");
        let stdout = child.stdout.take().expect("Failed to open stdout");

        // Channels for communication between threads
        let (tx_in, rx_in): (Sender<String>, Receiver<String>) = mpsc::channel();
        let (tx_out, rx_out): (Sender<String>, Receiver<String>) = mpsc::channel();

        // Thread to handle writing to Python stdin
        thread::spawn(move || write_to_stdin(stdin, rx_in));

        // Thread to handle reading from Python stdout
        thread::spawn(move || read_from_stdout(stdout, tx_out));

        PythonProcessParser {
            path: path.to_string(),
            child,
            tx_in,
            rx_out,
        }
    }

    fn render_script_non_streaming(&self, text: &str, token_ids: &[usize]) -> String {
        let mut env = Environment::new();
        env.add_template("reasoning_parser_non_streaming", REASONING_PYTHON_TEMPLATE)
            .unwrap();
        let tmpl = env.get_template("reasoning_parser_non_streaming").unwrap();
        tmpl.render(context! { path => self.path, text => text, token_ids => token_ids })
            .unwrap()
    }

    pub fn render_script_streaming(path: &str) -> String {
        let mut env = Environment::new();
        env.add_template(
            "reasoning_parser_streaming",
            REASONING_PYTHON_TEMPLATE_STREAMING,
        )
        .unwrap();
        let tmpl = env.get_template("reasoning_parser_streaming").unwrap();
        tmpl.render(context! { path => path }).unwrap()
    }
}

impl Drop for PythonProcessParser {
    fn drop(&mut self) {
        // Try graceful termination first
        let _ = self.child.kill();
        // Wait for the process to actually terminate
        let _ = self.child.wait();
    }
}

impl ReasoningParser for PythonProcessParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        let script = self.render_script_non_streaming(
            text,
            &token_ids
                .iter()
                .map(|&id| id as usize)
                .collect::<Vec<usize>>(),
        );
        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(script)
            .output()
            .expect("Failed to execute python process");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: format!("Error executing python process: {}", stderr),
            };
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        let normal_text = lines.next().unwrap_or("").to_string();
        let reasoning_text = lines.next().unwrap_or("").to_string();

        ParserResult {
            normal_text,
            reasoning_text,
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        let token_ids_str = token_ids
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<usize>>()
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<String>>()
            .join(",");

        if self
            .tx_in
            .send(format!("{}],[{}\n", text, token_ids_str))
            .is_err()
        {
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: String::from("Error: Python process communication failed"),
            };
        }

        if let Ok(normal_text) = self.rx_out.recv()
            && let Ok(reasoning_text) = self.rx_out.recv()
        {
            return ParserResult {
                normal_text,
                reasoning_text,
            };
        }

        ParserResult {
            normal_text: String::new(),
            reasoning_text: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_process_parser() {
        let mut parser = PythonProcessParser::new(
            "../../lib/bindings/python/src/dynamo/reasoning_parser/basic_parser.py",
        );
        let result = parser.detect_and_parse_reasoning(
            "<think>reasoning content</think> normal text.",
            &[1, 2, 3, 4],
        );
        // assert_eq!(result.normal_text, " normal text.");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_python_process_parser_streaming_in_chunks() {
        let mut parser = PythonProcessParser::new(
            "../../lib/bindings/python/src/dynamo/reasoning_parser/basic_parser.py",
        );
        let chunk1 = "<think>reasoning content part 1 ";
        let chunk2 = "reasoning content part 2";
        let chunk3 = "part 3</think> normal text.";
        let result = parser.parse_reasoning_streaming_incremental(chunk1, &[1, 2, 3, 4]);

        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "reasoning content part 1 ");
        let result2 = parser.parse_reasoning_streaming_incremental(chunk2, &[1, 2, 3, 4]);
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "reasoning content part 2");

        let result3 = parser.parse_reasoning_streaming_incremental(chunk3, &[1, 2, 3, 4]);
        assert_eq!(result3.normal_text, " normal text.");
        assert_eq!(result3.reasoning_text, "part 3");
    }
}
