%include "lib\util.asm"
.386

            section     .data
resident_start:
args_path   equ         0x82
args_len    equ         0x20
video       equ         0xb800
video_len   equ         80 * 25
buffer_len  equ         video_len + 25          ; +25 is for storing '\n'

; used for start and stop resident
active_key  equ         0x3b

; used for identify already loaded
finerprint  equ         1337

; determine start and end of grabber work
is_active   db          0

loaded_msg  db          "Grabber is already loaded in memory", 0x0a, 0x0d, '$'
init_msg    db          "Grabber is loaded as resident program", 0x0a, 0x0d, '$'
open_msg    db          "Failed to open target file!", 0x0a, 0x0d, '$'
write_msg   db          "Failed to write to file!", 0x0a, 0x0d, '$'
start_msg   db          "Grabber is start working...", 0x0a, 0x0d, '$'
end_msg     db          "Grabber is end working", 0x0a, 0x0d, '$'

            section     .bss
file_path   resb        args_len
file_id     resw        1

old_int
old_int_ip  resw        1
old_int_cs  resw        1

buffer      resb        buffer_len

            section     .text
start:      jmp         init

            nop
            nop

            section     .text
new_int:
            pushf

            call far    [cs:old_int]

            pusha
            push        ds
            push        es

            push        cs
            pop         ds

            mov         ah, 0x01                ; get entered key
            int         0x16
            jz          .int_end

            cmp         ah, active_key
            jz          .control
            jmp         .add_char

.control:   cmp         [is_active], byte 0
            jz          .activate
            jmp         .deactivate

.activate:  PRINT_STR   start_msg
            mov         [is_active], byte 1

            mov         ah, 0x3d                ; try to open file
            mov         al, 0x01
            xor         cl, cl
            mov         dx, args_path
            int         0x21
            jc          .open_err

            mov         [file_id], ax

            mov         ah, 0x40                ; trunc file
            mov         bx, [file_id]
            mov         cx, 0
            int         0x21

            push        video
            pop         es

            ; load screen frame to file
            xor         di, di
            xor         bx, bx
            mov         cx, 1
.loop:      push        cx                      ; save screen frame to buffer

            mov         ah, es:[di]
            mov         [buffer + bx], ah

            xor         dx, dx                  ; is end of line?
            mov         ax, cx
            mov         cx, 80
            div         cx
            cmp         dx, 0
            jnz         .skip

            inc         bx                      ; add new line symbol
            mov         [buffer + bx], byte 0x0a

.skip:      add         di, 2
            inc         bx
            
            pop         cx
            cmp         cx, video_len
            jz          .save
            inc         cx
            jmp         .loop

.save:      mov         ah, 0x40                ; save screen frame to file
            mov         bx, [file_id]
            mov         cx, buffer_len
            mov         dx, buffer
            int         0x21
            jc          .write_err

            jmp         .int_end

.open_err:  PRINT_STR   open_msg
            jmp         .int_end

.write_err: PRINT_STR   write_msg
            jmp         .int_end

.deactivate:PRINT_STR   end_msg
            mov         [is_active], byte 0

            mov         ah, 0x3e                ; close file
            mov         bx, [file_id]
            int         0x21

            jmp         .int_end

.add_char:  mov         [buffer], al            ; save ASCII character
            mov         ah, 0x40
            mov         bx, [file_id]
            mov         cx, 1
            mov         dx, buffer
            int         0x21

.int_end:   pop         es
            pop         ds
            popa
            iret
resident_end:

init:       call        save_args

            push        es

            mov         ah, 0x35                ; save old interrupt cs:ip
            mov         al, 0x09
            int         0x21

            mov         ax, [es:bx - 2]         ; save fingerprint
            mov         [cs:old_int_ip], bx
            mov         [cs:old_int_cs], es

            pop         es

            mov         [buffer], ax
            PRINT_INT   buffer
            PRINT_NEW_LINE
            cmp         [buffer], word finerprint
            jz          .loaded

            ; load resident fingerpring
            mov         [cs:new_int - 2], word finerprint

            mov         ah, 0x25                ; set new interrupt cs:ip
            mov         al, 0x09
            push        cs
            pop         ds
            mov         dx, new_int
            int         0x21

            push        es

            mov         ah, 0x35                ; save old interrupt cs:ip
            mov         al, 0x09
            int         0x21

            mov         ax, [es:bx - 2]
            mov         [buffer], ax

            pop         es

            PRINT_INT   buffer
            PRINT_NEW_LINE

            PRINT_STR   init_msg

            mov         ax, 0x3100              ; save program as resident
            mov         dx, resident_end
            sub         dx, resident_start
            add         dx, 0x100
            shr         dx, 4
            int         0x21

.loaded:    PRINT_STR   loaded_msg              ; resident program already in memory
            mov         ax, 0x4c01
            int         0x21

save_args:
            mov         cx, args_len
            mov         di, file_path
            mov         si, args_path
            cld
            rep movsb
            ret
