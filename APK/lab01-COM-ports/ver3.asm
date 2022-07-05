.model small
.stack 100h

.data

COM1    equ     3F8h
COM2    equ     2F8h

.code

start:
	; ds init
	mov ax, @data
	mov ds, ax

	; write 'g' into COM1
	mov dx, COM1
	mov al, 'g'
	out dx, al

	; reset al
	mov al, 00h

	; read symbol from COM2
	mov dx, COM2
	in  al, dx

	; print al
	mov ah, 02h
	mov dl, al
	int 21h

exit:
	mov ax, 4C00h
	int 21h

end start
