from __future__ import annotations


JSON_ACTION_GBNF = r'''
root ::= action
action ::= "{" ws "\"type\"" ws ":" ws action-type ws "," ws "\"args\"" ws ":" ws object ws "}"
action-type ::= "\"open_docs\"" | "\"search_web\"" | "\"read_file\"" | "\"grep\"" | "\"list_tree\"" | "\"write_file\"" | "\"delete_file\"" | "\"run_tests\"" | "\"run_command\"" | "\"open_browser\"" | "\"propose_patch\"" | "\"sandbox_patch\"" | "\"apply_patch\"" | "\"summarize_diff\"" | "\"finish\""
object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair ::= string ws ":" ws value
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
value ::= object | array | string | number | "true" | "false" | "null"
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws ::= [ \t\n\r]*
hex ::= [0-9a-fA-F]
'''


def build_agent_action_json_grammar() -> str:
    return JSON_ACTION_GBNF.strip()
