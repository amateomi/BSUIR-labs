%include "lib\util.asm"

; Useful data #################################################################
            section     .data
video       equ         0xb800

width       equ         80
height      equ         25

%define border          '#', 0b0_001_0_010
%define field           '.', 0b0_000_0_010
apple       db          '@', 0b0_000_0_100
snake       db          'o', 0b0_000_0_011

map
times width db          border
%rep height - 2
            db          border
times width - 2 db       field
            db          border
%endrep
times width db          border
map_size    equ         $ - map

top_border
%assign pos (0)
%rep width
            dw          pos
    %assign pos (pos + 2)
%endrep

left_border
%assign pos (width * 2)
%rep height - 2
            dw          pos
    %assign pos (pos + width * 2)
%endrep

right_border
%assign pos (width * 2 * 2 - 2)
%rep height - 2
            dw          pos
    %assign pos (pos + width * 2)
%endrep

down_border
%assign pos (width * 2 * (height - 1))
%rep width
            dw          pos
    %assign pos (pos + 2)
%endrep

snake_len   dw          1
last_dir    db          'w'
apple_exist dw          0

loss_msg    db          "Game is over!", 0x0a, 0x0d, '$'
score_msg   db          "Score $"
score_len   equ         20

delay_ms    dw          0xffff
delay_step  equ         0x0300

            section     .bss
snake_body  resw        width * height
apple_pos   resw        1
snake_pos   resw        1

map_buf     resw        map_size
; Main logic ##################################################################
            section     .text
start:      call        init
.game_loop: call        update_map
            call        delay
            call        handle_input

            call        is_apple
            jz          .grow
            call        is_bite
            jz          .game_over
            jmp         .shift

.grow:      call        grow_snake
.shift:     call        shift_snake_body

            jmp         .game_loop

.game_over: call        clear_screen
            PRINT_STR   loss_msg
            PRINT_STR   score_msg
            PRINT_INT   snake_len
            PRINT_NEW_LINE
            mov         ax, 0x4c00
            int         0x21
; Procedures ##################################################################
init:       
            xor         ah, ah                  ; set video-mode
            mov         al, 0x03                ; to 80x25
            int         0x10                    ; colored

            mov         ah, 0x01                ; off cursor
            mov         ch, 0x20
            int         0x10

            ; init snake start position
            mov         [snake_body], word map_size / 2
            ret

update_map:
            push        word [snake_body]       ; set snake_pos
            pop         word [snake_pos]

            cmp         [apple_exist], word 0
            jz          .gen_apple
            jmp         .print

.gen_apple: call        generate_apple
            mov         [apple_exist], word 1

.print:     call        print_buf_map
            call        print_buf_snake
            call        print_buf_apple
            call        buf_to_video
            call        print_score
            ret

clear_screen:
            mov         ax, 0x0002
            int         0x10
            ret

delay:
            mov         ah, 0x86
            mov         cx, 0x0001
            mov         dx, [delay_ms]
            int         0x15
            ret

handle_input:
            mov         ah, 0x01                ; check input
            int         0x16
            jz          .old_dir
            xor         ah, ah                  ; get new direction
            int         0x16
            jmp         .cmp_dir

.old_dir:   mov         al, [last_dir]

.cmp_dir:   cmp         al, 'w'
            jz          .up
            cmp         al, 'a'
            jz          .left
            cmp         al, 's'
            jz          .down
            cmp         al, 'd'
            jz          .right
            ; any input except wasd is count as not input at all
            jmp         .old_dir

.up:        cmp         [last_dir], byte 's'
            jz          .old_dir
            sub         [snake_body], word width * 2
            jmp         .save_dir

.left:      cmp         [last_dir], byte 'd'
            jz          .old_dir
            sub         [snake_body], word 2
            jmp         .save_dir

.down:      cmp         [last_dir], byte 'w'
            jz          .old_dir
            add         [snake_body], word width * 2
            jmp         .save_dir

.right:     cmp         [last_dir], byte 'a'
            jz          .old_dir
            add         [snake_body], word 2
            jmp         .save_dir

.save_dir:  mov         [last_dir], al

            ; check snake on border, move to the other side when true
            mov         ax, [snake_body]
            call        is_top_border
            jz          .top_brdr
            call        is_left_border
            jz          .left_brdr
            call        is_right_border
            jz          .right_brdr
            call        is_down_border
            jz          .down_brdr
            jmp         .end

.top_brdr:  add         [snake_body], word width * (height - 2) * 2
            jmp         .end

.left_brdr: add         [snake_body], word (width - 2) * 2
            jmp         .end

.right_brdr:sub         [snake_body], word (width - 2) * 2
            jmp         .end

.down_brdr: sub         [snake_body], word width * (height - 2) * 2

.end:       ret

is_top_border: ; set zf
            mov         cx, width
            mov         di, top_border
            cld
            repnz scasw
            ret

is_left_border: ; set zf
            mov         cx, height - 2
            mov         di, left_border
            cld
            repnz scasw
            ret

is_right_border: ; set zf
            mov         cx, height - 2
            mov         di, right_border
            cld
            repnz scasw
            ret

is_down_border: ; set zf
            mov         cx, width
            mov         di, down_border
            cld
            repnz scasw
            ret
; Printing ####################################################################
print_buf_map:
            mov         cx, map_size
            mov         di, map_buf
            mov         si, map
            cld
            rep movsb
            ret

print_buf_snake:
            mov         cx, [snake_len]
.loop:      ; bx = cx * 2 - 2
            mov         bx, cx
            shl         bx, 1
            sub         bx, 2

            mov         ax, [snake]
            mov         bx, [snake_body + bx]
            mov         [map_buf + bx], ax
            loop        .loop
            ret

print_buf_apple:
            mov         ax, [apple]
            mov         bx, [apple_pos]
            mov         [map_buf + bx], ax
            ret

buf_to_video:
            push        es

            push        video                   ; load video segment in es
            pop         es

            mov         cx, map_size - score_len
            mov         di, score_len
            mov         si, map_buf + score_len
            cld
            rep movsb

            pop         es
            ret

print_score:
            mov         ah, 0x02                ; set cursor to start
            xor         bh, bh
            xor         dx, dx
            int         0x10
            PRINT_STR   score_msg
            PRINT_INT   snake_len
            ret

; Apple #######################################################################
generate_apple:
            mov         ah, 0x02                ; save time into cx:dx
            int         0x1a

            mov         ax, dx
            xor         dx, dx
            mov         cx, width - 2
            div         cx                      ; dx is random apple x
            shl         dx, 1                   ; convert apple x into offset

            push        dx                      ; save apple x on stack

            mov         ah, 0x02                ; save time into cx:dx
            int         0x1a

            mov         ax, dx
            xor         dx, dx
            mov         cx, height - 2
            div         cx                      ; dx is random apple y
            mov         ax, dx                  ; convert apple y into offset
            xor         dx, dx
            mov         cx, width * 2
            mul         cx                      ; result in ax

            ; skip first line
            mov         [apple_pos], word (width + 1) * 2

            add         [apple_pos], ax

            pop         ax                      ; load apple x from stack
            add         [apple_pos], ax

            ret

is_apple: ; set zf
            mov         bx, [snake_body]
            mov         ax, [map_buf + bx]
            cmp         ax, word [apple]
            ret
; Snake #######################################################################
grow_snake:
            mov         ax, [snake_pos]
            mov         bx, [snake_len]
            shl         bx, 1
            mov         [snake_body + bx], ax
            inc         word [snake_len]

            mov         [apple_exist], word 0
            sub         [delay_ms], word delay_step
            ret

shift_snake_body:
            mov         bx, 2
            mov         cx, [snake_len]
            mov         dx, [snake_pos]

.loop:      push        word [snake_body + bx]
            mov         [snake_body + bx], dx
            pop         dx
            add         bx, 2
            loop        .loop
            ret

is_bite: ; set zf
            mov         bx, [snake_body]
            mov         ax, [map_buf + bx]
            cmp         ax, word [snake]
            ret
