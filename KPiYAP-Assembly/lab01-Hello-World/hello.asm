            BITS    16
            org     0x100

            section .text
start:      mov     dx, message
            mov     ah, 0x09
            int     0x21
            ret

            section .data
message     db      "Hello world!", 0x0d, 0x0a, '$'
