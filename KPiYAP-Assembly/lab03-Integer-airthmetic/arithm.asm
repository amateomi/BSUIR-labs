%include "lib\util.asm"

%macro OP_TWO_ARG 1 ; %1 is operation name
            mov         ax, [num1]
            %1          ax, [num2]
            mov         [result], ax
            PRINT_STR   %1_msg
            PRINT_INT   result
            PRINT_STR   bin_msg
            PRINT_BIN   result
            PRINT_NEW_LINE
%endmacro

%macro OP_ONE_ARG 1 ; %1 is operation name
            xor         dx, dx
            mov         ax, [num1]
            %1          word [num2]
            push        ax
            push        dx
            PRINT_STR   %1_msg
            PRINT_STR   dx_msg
            pop         word [result]
            PRINT_INT   result
            PRINT_STR   ax_msg
            pop         word [result]
            PRINT_INT   result
            PRINT_NEW_LINE
%endmacro

%macro SHIFT 1 ; %1 is some shift operation
            mov         ax, [num1]
            mov         cx, [num2]
            %1          ax, cl                  ; used only 5 bits in cl
            mov         [result], ax
            PRINT_STR   %1_msg
            PRINT_INT   result
            PRINT_STR   bin_msg
            PRINT_BIN   result
            PRINT_NEW_LINE
%endmacro

            section     .bss
num1        resw        1
num2        resw        1
result      resw        1

            section     .data
input_int   db          "Input 16 bit integer: $"
div_zero    db          "Division by zero is not allowed!", 0x0d, 0x0a, '$'

add_msg     db          "add: $"
adc_msg     db          "adc: $"
sub_msg     db          "sub: $"
sbb_msg     db          "sbb: $"

mul_msg     db          "mul: $"
imul_msg    db          "imul: $"
div_msg     db          "div: $"
idiv_msg    db          "idiv: $"

and_msg     db          "and: $"
or_msg      db          "or: $"
xor_msg     db          "xor: $"
not_msg     db          "not: $"

shr_msg     db          "shr: $"
shl_msg     db          "shl: $"
sar_msg     db          "sar: $"
sal_msg     db          "sal: $"

dx_msg      db          "dx=$"
ax_msg      db          " ax=$"
entered_msg db          "Entered=$"
bin_msg     db          " bin: $"
error_msg   db          "Invalid input!", 0x0d, 0x0a, '$'

            section     .text
start:      PRINT_STR   input_int
            INPUT_INT   num1
            cmp         dx, 0
            jz          error
            PRINT_NEW_LINE

            PRINT_STR   entered_msg
            PRINT_INT   num1
            PRINT_STR   bin_msg
            PRINT_BIN   num1
            PRINT_NEW_LINE

            PRINT_STR   input_int
            INPUT_INT   num2
            cmp         dx, 0
            jz          error
            PRINT_NEW_LINE

            PRINT_STR   entered_msg
            PRINT_INT   num2
            PRINT_STR   bin_msg
            PRINT_BIN   num2
            PRINT_NEW_LINE

            PRINT_NEW_LINE

            OP_TWO_ARG  add
            OP_TWO_ARG  adc
            OP_TWO_ARG  sub
            OP_TWO_ARG  sbb
            OP_ONE_ARG  mul
            OP_ONE_ARG  imul

            cmp         [num2], word 0
            jz          skip_div
            OP_ONE_ARG  div
            OP_ONE_ARG  idiv
            jmp         after_div

skip_div:   PRINT_STR   div_zero
after_div:

            OP_TWO_ARG  and
            OP_TWO_ARG  or
            OP_TWO_ARG  xor
            ; not
            PRINT_STR   not_msg
            mov         ax, [num1]
            mov         [result], ax
            not         word [result]
            PRINT_INT   result
            PRINT_STR   bin_msg
            PRINT_BIN   result
            PRINT_NEW_LINE

            SHIFT       shr
            SHIFT       shl
            SHIFT       sar
            SHIFT       sal

            jmp         end

error:      PRINT_NEW_LINE
            PRINT_STR   error_msg
            jmp         end

end:        mov         ax, 0x4c00
            int         0x21
