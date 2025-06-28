// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;

use clap::Parser;
use clap::CommandFactory;

use dynamo_llm::entrypoint::input::Input;
use dynamo_run::Output;
use dynamo_runtime::logging;

const HELP: &str = r#"
dynamo-run is a single binary that wires together the various inputs (http, text, network) and workers (network, engine), that runs the services. It is the simplest way to use dynamo locally.

Verbosity:
- -v enables debug logs
- -vv enables full trace logs
- Default is info level logging

Example:
- cargo build --features cuda -p dynamo-run
- cd target/debug
- ./dynamo-run Qwen/Qwen3-0.6B
- OR: ./dynamo-run /data/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
"#;


fn main() -> anyhow::Result<()> {
    // Set log level based on verbosity flag
    let log_level = match dynamo_run::Flags::try_parse() {
        Ok(flags) => match flags.verbosity {
            0 => "info",
            1 => "debug",
            2 => "trace",
            _ => {
                return Err(anyhow::anyhow!(
                    "Invalid verbosity level. Valid values are v (debug) or vv (trace)"
                ))
            }
        },
        Err(_) => "info",
    };

    if log_level != "info" {
        std::env::set_var("DYN_LOG", log_level);
    }

    logging::init();

    // max_worker_threads and max_blocking_threads from env vars or config file.
    let rt_config = dynamo_runtime::RuntimeConfig::from_settings()?;

    // One per process. Wraps a Runtime with holds two tokio runtimes.
    let worker = dynamo_runtime::Worker::from_config(rt_config)?;

    worker.execute(wrapper)
}

async fn wrapper(runtime: dynamo_runtime::Runtime) -> anyhow::Result<()> {
    let mut in_opt = None;
    let mut out_opt = None;
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty()
        || args[0] == "-h"
        || args[0] == "--help"
        || (args.iter().all(|arg| arg == "-v" || arg == "-vv"))
    {
        let engine_list = Output::available_engines().join("|");
        let help_message = dynamo_run::Flags::command().render_help().to_string();
        let usage = help_message.replace("ENGINE_LIST", &engine_list);
        println!("{usage}");
        println!("{HELP}");
        return Ok(());
    } else if args[0] == "--version" {
        if let Some(describe) = option_env!("VERGEN_GIT_DESCRIBE") {
            println!("dynamo-run {}", describe);
        } else {
            println!("Version not available (git describe not available)");
        }
        return Ok(());
    }
    for arg in env::args().skip(1).take(2) {
        let Some((in_out, val)) = arg.split_once('=') else {
            // Probably we're defaulting in and/or out, and this is a flag
            continue;
        };
        match in_out {
            "in" => {
                in_opt = Some(val.try_into()?);
            }
            "out" => {
                out_opt = Some(val.try_into()?);
            }
            _ => {
                let help_message = dynamo_run::Flags::command().render_help();
                anyhow::bail!("Invalid argument, must start with 'in' or 'out. {help_message}");
            }
        }
    }
    let mut non_flag_params = 1; // binary name
    let in_opt = match in_opt {
        Some(x) => {
            non_flag_params += 1;
            x
        }
        None => Input::default(),
    };
    if out_opt.is_some() {
        non_flag_params += 1;
        println!("out_opt is some");
    }
    println!("in: {in_opt}");
    //println!("{out_opt}");
    println!("non_flags_params: {non_flag_params}");

    // Clap skips the first argument expecting it to be the binary name, so add it back
    // Note `--model-path` has index=1 (in lib.rs) so that doesn't need a flag.
    let flags = dynamo_run::Flags::try_parse_from(
        ["dynamo-run".to_string()]
            .into_iter()
            .chain(env::args().skip(non_flag_params)),
    )?;
    let chain_flags = env::args().skip(non_flag_params).collect::<Vec<_>>().join(" ");;
    println!("chain flags: {chain_flags}");

    if is_in_dynamic(&in_opt) && is_out_dynamic(&out_opt) {
        anyhow::bail!("Cannot use endpoint for both in and out");
    }

    dynamo_run::run(runtime, in_opt, out_opt, flags).await
}

fn is_in_dynamic(in_opt: &Input) -> bool {
    matches!(in_opt, Input::Endpoint(_))
}

fn is_out_dynamic(out_opt: &Option<Output>) -> bool {
    matches!(out_opt, Some(Output::Dynamic))
}
