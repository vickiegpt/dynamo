import sys
import os

PATH = "./lib/bindings/python/examples/basic_reasoning_parser/basic_parser.py"

# get absolute path
PATH = os.path.abspath(PATH)


#extract parent directory
parent_dir = os.path.dirname(PATH)
print(f"Parent directory: {parent_dir}")

#extract filename without extension
module_name, _ = os.path.splitext(os.path.basename(PATH))

sys.path.append(parent_dir)

#import the module
reasoning_module = __import__(module_name)

# find class with BaseReasoningParser in its bases

cls_name = ""

for name, obj in vars(reasoning_module).items():
    if hasattr(obj, "__bases__") and "BaseReasoningParser" in [base.__name__ for base in obj.__bases__]:
        print(f"Found class with BaseReasoningParser in its bases: {name}")
        cls_name = name
        break

ReasoningParserClass = getattr(reasoning_module, cls_name)

parser_instance = ReasoningParserClass()

normal_text, reasoning_text = parser_instance.parse_reasoning_streaming_incremental("<think>reasoning content</think> normal text.", [])

print(f"Normal Text: '{normal_text}'")
print(f"Reasoning Text: '{reasoning_text}'")
