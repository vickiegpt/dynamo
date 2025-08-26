// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use rustpython_parser::{
    Mode,
    ast::{Constant, Expr, Mod},
    parse,
};
use serde_json::{Value, json};

fn strip_text(message: &str) -> String {
    // Remove unexpected python tags if any
    message
        .replace("<|python_start|>", "")
        .replace("<|python_end|>", "")
}

fn get_regex_matches(message: &str) -> Vec<String> {
    use regex::Regex;
    let pattern = r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*?,\s*)*([a-zA-Z]+\w*=.*?\s?)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*?,\s*)*([a-zA-Z]+\w*=.*?\s*)?\)\s*)+\]";
    let re = Regex::new(pattern).unwrap();

    let mut matches = Vec::new();
    for cap in re.find_iter(message) {
        matches.push(cap.as_str().to_string());
    }
    matches
}

pub fn parse_tool_calls(src: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
    let ast = parse(src, Mode::Expression, "<input>").unwrap();

    /*
    AST: Expression(ModExpression {
        range: (),
        body: List(ExprList {
            range: 0..25,
            elts: [Call(...), Call(...)]
            ctx: Load
        })
    })
    */
    let body = match ast {
        Mod::Expression(mod_expr) => mod_expr.body,
        _ => return Ok(vec![]),
    };

    let elts = match *body {
        Expr::List(expr_list) => expr_list.elts,
        _ => return Ok(vec![]),
    };

    let mut res = Vec::new();
    for (idx, elt) in elts.iter().enumerate() {
        let (func, keywords) = match elt {
            Expr::Call(call) => (&call.func, &call.keywords),
            _ => continue,
        };

        let name = match func.as_ref() {
            Expr::Name(name) => name.id.clone(),
            _ => continue,
        };

        let mut obj = serde_json::Map::new();
        for keyword in keywords.iter() {
            let arg_name = keyword
                .arg
                .as_ref()
                .ok_or("**kwargs not allowed")
                .unwrap()
                .to_string();
            obj.insert(arg_name, const_expr(&keyword.value).unwrap());
        }

        res.push(ToolCallResponse {
            id: format!("call-{}", idx + 1),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: serde_json::to_string(&Value::Object(obj)).unwrap(),
            },
        });
    }
    Ok(res)
}

// constants only: int/float/str/bool/None
fn const_expr(e: &Expr) -> Result<Value, Box<dyn std::error::Error>> {
    // TODO: Add support for lists/dicts
    match e {
        Expr::Constant(constant) => Ok(match &constant.value {
            Constant::Bool(b) => json!(b),
            Constant::None => Value::Null,
            Constant::Int(i) => json!(i.to_string()),
            Constant::Float(f) => json!(f),
            Constant::Str(s) => json!(s),
            _ => return Err("unsupported constant type".into()),
        }),
        _ => Err("only constant values are allowed".into()),
    }
}

pub fn try_tool_call_parse_pythonic(
    message: &str,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let stripped = strip_text(message).trim().to_string();

    // Early exit if no content
    if stripped.is_empty() {
        return Ok((vec![], Some(String::new())));
    }

    let matches = get_regex_matches(&stripped);
    if matches.is_empty() {
        return Ok((vec![], Some(stripped)));
    }

    println!("Matches: {:?}", matches[0]);

    let tool_response = parse_tool_calls(&matches[0]);

    Ok((tool_response?, Some(String::new()))) // TODO: Add support for normal text 
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[test]
    fn test_get_regex_matches() {
        let message = "[foo(a=1, b=2), bar(x=3)]";
        let (result, _) = try_tool_call_parse_pythonic(message).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone()); // TODO: Add support for normal text 
        assert_eq!(name, "foo");
        assert_eq!(args["a"], "1");
        assert_eq!(args["b"], "2");
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "bar");
        assert_eq!(args["x"], "3");
    }
}
