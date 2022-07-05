.model small
.stack 100h

.data

COM1  equ      0
COM2  equ      1

.code

start:
    ; ds init
    mov ax, @data
    mov ds, ax

    ; write 'g' into COM1
    mov ah, 01h
    mov al, 'g'
    mov dx, COM1
    int 14h

    ; reset al
    mov al, 00h

    ; read symbol from COM2
    mov ah, 02h
    mov dx, COM2
    int 14h

    ; print al
    mov ah, 02h
    mov dl, al
    int 21h

exit:
    mov ax, 4C00h
    int 21h

end start
