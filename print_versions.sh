#!/bin/bash

main_hash=$(git rev-parse HEAD)
main_branch=$(git rev-parse --abbrev-ref HEAD)
sub_status=$(git submodule status)

code_block="\`\`\`"

msg="\
$code_block
Project version: $main_hash ($main_branch)
Subprojects:
$sub_status
$code_block\
"

echo "$msg"

