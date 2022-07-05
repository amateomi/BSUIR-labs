%include "lib\util.asm"

            section     .data
video       equ         0xb800
width       equ         80
height      equ         25
size        equ         width * height

background
%rep height - 1
    %rep width / 2
        %assign back (0b0_010_0_000)
        %rep 2
            db          ' ', back
            db          ' ', back
            %assign back (back + 0b0_001_0_000)
        %endrep
    %endrep
%endrep
back_size   equ         $-background

file_path   equ         0x82

file_pos    dw          0
is_end      dw          0

chunk_size  equ         size / 2
hex_buf 
times size  db          ' '

last_read   dw          chunk_size

open_msg    db          "Failed to open file!", 0x0a, 0x0d, '$'

            section     .bss
file_id     resw        1
chunk       resb        chunk_size

            section     .text
start:      call        init
            mov         ah, 0x3d                ; try to open file
            mov         al, 0x00
            mov         dx, file_path
            int         0x21
            jc          .open_err
            mov         [file_id], ax

.read:      mov         ah, 0x3f                ; read chunk from file
            mov         bx, [file_id]
            mov         cx, chunk_size
            mov         dx, chunk
            int         0x21

            mov         [last_read], ax         ; ax is amount of read bytes

            cmp         ax, chunk_size          
            jz          .not_end
            jmp         .end

.end:       mov         [is_end], word 1
            jmp         .print
.not_end:   mov         [is_end], word 0

.print:     call        print_screen

.input:     mov         ah, 0x01
            int         0x16
            jz          .input
            xor         ah, ah
            int         0x16
            cmp         al, 'w'
            jz          .up
            cmp         al, 's'
            jz          .down
            cmp         al, 'q'
            jz          .exit
            jmp         .input

.up:        cmp         [file_pos], word 0
            jz          .input
            dec         word [file_pos]

            mov         ah, 0x42                ; set pos in file
            mov         al, 0x01
            mov         bx, [file_id]
            mov         cx, 0xffff
            mov         dx, [last_read]
            neg         dx
            sub         dx, chunk_size
            int         0x21
            jmp         .read

.down:      cmp         [is_end], word 1
            jz          .input
            inc         word [file_pos]

            jmp         .read

.open_err:  mov         ax, 0x0002              ; clear screen
            int         0x10
            PRINT_STR   open_msg
            mov         ax, 0x4c00
            int         0x21

.exit:      PRINT_NEW_LINE
            mov         ah, 0x3e                ; close file
            mov         bx, file_id
            int         0x21

            mov         ax, 0x0002              ; clear screen
            int         0x10

            mov         ax, 0x4c00
            int         0x21

init:       
            xor         ah, ah                  ; set video-mode
            mov         al, 0x03                ; to 80x25
            int         0x10                    ; colored

            mov         ah, 0x01                ; off cursor
            mov         ch, 0x20
            int         0x10
            ret

clear_hex:
            mov         cx, size - 1
.loop:      mov         bx, cx
            mov         [hex_buf + bx], byte ' '
            loop        .loop
            ret

print_screen:
            push        es                      

            call        clear_hex

            mov         bx, ax                  ; prepare bx and di for converting
            dec         bx                      
            mov         di, bx
            shl         di, 1
.convert:   BYTE_TO_HEX chunk + bx, hex_buf + di
            dec         bx
            sub         di, 2
            cmp         bx, -1
            jnz         .convert

            push        video
            pop         es
            cld

            mov         cx, back_size           ; print background
            xor         di, di
            mov         si, background
            rep movsb

            xor         di, di                  ; print hex data
            xor         si, si
            mov         cx, size
.loop:      mov         al, [hex_buf + si]
            mov         [es:di], al
            add         di, 2
            inc         si
            loop        .loop

            pop         es
            ret
