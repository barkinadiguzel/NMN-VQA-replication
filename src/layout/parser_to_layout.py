from collections import namedtuple

ModuleNode = namedtuple("ModuleNode", ["type", "instance", "args"])

def parse_question_to_layout(question: str):
    q = question.lower()
    
    if "color" in q:
        return ModuleNode("classify", "color", [ModuleNode("attend", "tie", [])])
    elif "is there" in q:
        # Example: "Is there a red circle above a square?"
        red_node = ModuleNode("attend", "red", [])
        circle_node = ModuleNode("attend", "circle", [])
        above_node = ModuleNode("re_attend", "above", [circle_node])
        combine_node = ModuleNode("combine", "and", [red_node, above_node])
        return ModuleNode("measure", "is", [combine_node])
    else:
        return ModuleNode("attend", "object", [])
