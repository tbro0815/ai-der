#include <assert.h>
#include <stdio.h>

#include "../qwen38_control.h"

int main(void)
{
    assert(q38_control_action(COLI_SERVE_COMMAND_CANCEL, "req-7", "req-7") ==
           Q38_CONTROL_CANCEL);
    assert(q38_control_action(COLI_SERVE_COMMAND_STOP, "req-7", "req-7") ==
           Q38_CONTROL_STOP);
    assert(q38_control_action(COLI_SERVE_COMMAND_CANCEL, "req-8", "req-7") ==
           Q38_CONTROL_NOT_FOUND);
    assert(q38_control_action(COLI_SERVE_COMMAND_SUBMIT, "req-8", "req-7") ==
           Q38_CONTROL_BUSY);
    assert(q38_control_action(COLI_SERVE_COMMAND_RESET, "req-8", "req-7") ==
           Q38_CONTROL_BUSY);
    puts("qwen38 control tests: ok");
    return 0;
}
