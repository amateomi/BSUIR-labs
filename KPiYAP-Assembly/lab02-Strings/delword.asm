%include "lib\util.asm"

            section     .bss
string      resb        max_len
delete      resb        max_len

            section     .data
max_len     equ         200

input_str   db          "Input string (max=200): $"
input_del   db          "Input word to delete word in front of which (max=200): $"

result_msg  db          "Result=$"
entered_msg db          "Entered=$"

empty_msg   db          "Entered string is empty!$"
not_wrd_msg db          "Word must not contain spaces!$"

            section     .text
start:      PRINT_STR   input_str
            INPUT_STR   string, max_len
            PRINT_NEW_LINE

            STR_LEN     string
            cmp         cx, 0
            jz          empty

            PRINT_STR   entered_msg
            PRINT_STR   string
            PRINT_NEW_LINE

            PRINT_STR   input_del
            INPUT_STR   delete, max_len
            PRINT_NEW_LINE

            STR_LEN     delete
            cmp         cx, 0
            jz          empty

            call        is_word
            jz          not_word

            PRINT_STR   entered_msg
            PRINT_STR   delete
            PRINT_NEW_LINE

            call        delete_word_before_given

            PRINT_STR   result_msg
            PRINT_STR   string
            PRINT_NEW_LINE
            ret

empty:      PRINT_STR   empty_msg
            PRINT_NEW_LINE
            ret

not_word:   PRINT_STR   not_wrd_msg
            PRINT_NEW_LINE
            ret

is_word: ; check delete (buffer in .bss) for spaces, set zf when space found
            push        di

            STR_LEN     delete
            mov         al, ' '
            mov         di, delete
            repnz scasb

            pop         di
            ret

skip_spaces: ; skip spaces using di, erases cx and al, at the end di point on first non space
            mov         cx, 0xffff
            mov         al, ' '
            repz scasb
            dec         di
            ret

skip_word: ; skip word using di, set al to 1 when word is string end and 0 when not
            xor         al, al
.loop:      cmp         [di], byte '$'
            jz          .str_end
            cmp         [di], byte ' '
            jz          .end

            inc         di
            jmp         .loop

.str_end:   inc         al                      ; set 1 to demonstrate that word is string end
.end:       inc         di                      ; move to ' ' or '$'
            ret

delete_word_before_given:
            cld
            mov         di, string
            call        skip_spaces

.loop:      mov         si, di                  ; save first word start

            call        skip_word               ; set al to 1 when word is last in string
            cmp         al, 1
            jz          .exit

            call        skip_spaces

            CMP_WRD     di, delete
            jnz         .loop

            push        si                      ; save first word position

            STR_LEN     di                      ; store len in cx
            inc         cx                      ; must move '$' too

            xchg        si, di
            rep movsb

            pop         di                      ; load deleted word position

            jmp         .loop
.exit:      ret
