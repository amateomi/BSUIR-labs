%include "lib\util.asm"

            section     .data
source_path equ         0x82
cmd_max_len equ         0x80

target_path db          "C:\view.com", 0

; Exec Parameter Block
EPB         dw          0x0000                  ; save current environment
            dw          commandline, 0          ; command name
            dw          0x005C, 0, 0x006C, 0    ; FCB

open_msg    db          "Failed to open source file!", 0x0a, 0x0d, '$'

            section     .bss
file_id     resw        1

commandline resb        cmd_max_len

            section     .text
start:      call        read_args

            ; move stack pointer for next program
            mov         sp, program_len + 0x100

            ; free memory
            mov         ah, 0x4a                
            mov         bx, program_len + 0x100
            shr         bx, 4
            inc         bx
            int         0x21

            ; load cs in EPB for next program
            mov         ax, cs
            mov         [EPB + 4], ax
            mov         [EPB + 8], ax
            mov         [EPB + 12], ax

            ; execute next program
            mov         ax, 0x4b00
            mov         bx, EPB
            mov         dx, target_path
            int         0x21

            mov         ax, 0x4c00
            int         0x21

read_args:
            mov         ah, 0x3d                ; try to open file
            mov         al, 0x00
            mov         dx, source_path
            int         0x21
            jc          .open_err
            mov         [file_id], ax

            mov         ah, 0x3f                ; read arg from file
            mov         bx, [file_id]
            mov         cx, cmd_max_len
            mov         dx, commandline + 2
            int         0x21

            mov         [commandline], ax       ; save arg size

            mov         ah, 0x3e                ; close file
            mov         bx, file_id
            int         0x21

            ret

.open_err:
            PRINT_STR   open_msg
            mov         ax, 0x4c00
            int         0x21

program_len
