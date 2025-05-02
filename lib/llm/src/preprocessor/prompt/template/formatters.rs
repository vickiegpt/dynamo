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

use std::sync::Arc;

use super::tokcfg::{raise_exception, strftime_now, tojson, ChatTemplate};
use super::{ContextMixins, HfTokenizerConfigJsonFormatter, JinjaEnvironment};
use either::Either;
use minijinja::Environment;
use tracing;

impl JinjaEnvironment {
    fn env(self) -> Environment<'static> {
        self.env
    }
}

impl Default for JinjaEnvironment {
    fn default() -> Self {
        let mut env = Environment::new();

        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);

        JinjaEnvironment { env }
    }
}

impl HfTokenizerConfigJsonFormatter {
    pub fn new(config: ChatTemplate, mixins: ContextMixins) -> anyhow::Result<Self> {
        let mut env = JinjaEnvironment::default().env();

        let chat_template = match config.chat_template.as_ref() {
            Some(template) => template,
            None => {
                tracing::warn!("chat_template is not present in the tokenizer_config.json file.");
                env.add_template(
                    "default",
                    r#"{% if messages[0]['role'] == 'system' %}
        {% set loop_messages = messages[1:] %}
        {% set system_message = messages[0]['content'] %}
        {% else %}
        {% set loop_messages = messages %}
        {% set system_message = '' %}
        {% endif %}
        {% if system_message %}
        <s>{% if add_generation_prompt %}{{ bos_token }}{% endif %}{{ system_message }}

        {% endif %}
        {% for message in loop_messages %}
        {% if message['role'] == 'user' %}
        {{ user_token }}{{ message['content'] }}
        {% elif message['role'] == 'assistant' %}
        {{ assistant_token }}{{ message['content'] }}{% if loop.last and add_generation_prompt %}{{ eos_token }}{% endif %}
        {% endif %}
        {% endfor %}
        {% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}
        {{ assistant_token }}
        {% endif %}"#
                ).unwrap();
                return Ok(HfTokenizerConfigJsonFormatter {
                    env: env,
                    config,
                    mixins: Arc::new(mixins),
                    supports_add_generation_prompt: false, // Default behavior
                });
            }
        };

        // add pycompat
        // todo: should we use this: minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

        env.add_filter("tojson", tojson);

        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);

        let mut supports_add_generation_prompt = None;

        match &chat_template.0 {
            Either::Left(x) => {
                if x.contains("add_generation_prompt") {
                    tracing::debug!("Chat template contains `add_generation_prompt` key. This model supports add_generation_prompt.");
                    supports_add_generation_prompt = Some(true);
                }
                env.add_template_owned("default", x.to_string())?;
                env.add_template_owned("tool_use", x.to_string())?;
            }
            Either::Right(map) => {
                for t in map {
                    for (k, v) in t.iter() {
                        if v.contains("add_generation_prompt") {
                            match supports_add_generation_prompt {
                                Some(true) | None => {
                                    tracing::debug!("Chat template contains `add_generation_prompt` key. This model supports add_generation_prompt.");
                                    supports_add_generation_prompt = Some(true);
                                }
                                Some(false) => {
                                    tracing::warn!("Not all templates contain `add_generation_prompt` key. This model does not support add_generation_prompt.");
                                }
                            }
                        } else {
                            supports_add_generation_prompt = Some(false);
                        }
                        env.add_template_owned(k.to_string(), v.to_string())?;
                    }
                }
                if env.templates().count() == 0 {
                    anyhow::bail!("Chat template does not contain a `tool_use` or `default` key. Please ensure it contains at least a `default` key, although `tool_use` should be specified for using tools.");
                }
            }
        }

        Ok(HfTokenizerConfigJsonFormatter {
            env,
            config,
            mixins: Arc::new(mixins),
            supports_add_generation_prompt: supports_add_generation_prompt.unwrap_or(false),
        })
    }
}

// impl JinjaEnvironment {
//     /// Renders the template with the provided messages.
//     /// This function reuses the pre-compiled template for efficiency.
//     pub fn render(&self, template_id: &str, ctx: &dyn erased_serde::Serialize) -> Result<String> {
//         let tmpl = self.env.get_template(template_id)?;
//         Ok(tmpl.render(ctx)?)
//     }

//     // fn apply_tool_template()
// }
