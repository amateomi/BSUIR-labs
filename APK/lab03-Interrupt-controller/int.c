#include <dos.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER_BASE_VECTOR 0x08
#define SLAVE_BASE_VECTOR  0x98

void print_byte(char far* screen, unsigned char byte) {
  int bit;
  int i;
  for (i = 0; i < 8; ++i) {
    bit = byte % 2;
    byte = byte >> 1;
    *screen = '0' + bit;
    screen += 2;
  }
}

void print(void) {
  char far* screen = (char far*)MK_FP(0xB800, 0);

  // Master mask
  print_byte(screen, inp(0x21));
  screen += 18;
  // Slave mask
  print_byte(screen, inp(0xA1));

  screen += 142;

  // Master request
  outp(0x20, 0x0A);
  print_byte(screen, inp(0x20));
  screen += 18;
  // Slave request
  outp(0xA0, 0x0A);
  print_byte(screen, inp(0xA0));

  screen += 142;

  // Master service
  outp(0x20, 0x0B);
  print_byte(screen, inp(0x20));
  screen += 18;
  // Master service
  outp(0xA0, 0x0B);
  print_byte(screen, inp(0xA0));
}

// Master interruptions handlers ----------------------------------------------
void interrupt (*old_irq0_handler)(void);
void interrupt (*old_irq1_handler)(void);
void interrupt (*old_irq2_handler)(void);
void interrupt (*old_irq3_handler)(void);
void interrupt (*old_irq4_handler)(void);
void interrupt (*old_irq5_handler)(void);
void interrupt (*old_irq6_handler)(void);
void interrupt (*old_irq7_handler)(void);

void interrupt new_irq0_handler(void) { print(); old_irq0_handler(); }
void interrupt new_irq1_handler(void) { print(); old_irq1_handler(); }
void interrupt new_irq2_handler(void) { print(); old_irq2_handler(); }
void interrupt new_irq3_handler(void) { print(); old_irq3_handler(); }
void interrupt new_irq4_handler(void) { print(); old_irq4_handler(); }
void interrupt new_irq5_handler(void) { print(); old_irq5_handler(); }
void interrupt new_irq6_handler(void) { print(); old_irq6_handler(); }
void interrupt new_irq7_handler(void) { print(); old_irq7_handler(); }
//-----------------------------------------------------------------------------

// Slave interruptions handlers -----------------------------------------------
void interrupt (*old_irq8_handler)(void);
void interrupt (*old_irq9_handler)(void);
void interrupt (*old_irq10_handler)(void);
void interrupt (*old_irq11_handler)(void);
void interrupt (*old_irq12_handler)(void);
void interrupt (*old_irq13_handler)(void);
void interrupt (*old_irq14_handler)(void);
void interrupt (*old_irq15_handler)(void);

void interrupt new_irq8_handler(void) { print(); old_irq8_handler(); }
void interrupt new_irq9_handler(void) { print(); old_irq9_handler(); }
void interrupt new_irq10_handler(void) { print(); old_irq10_handler(); }
void interrupt new_irq11_handler(void) { print(); old_irq11_handler(); }
void interrupt new_irq12_handler(void) { print(); old_irq12_handler(); }
void interrupt new_irq13_handler(void) { print(); old_irq13_handler(); }
void interrupt new_irq14_handler(void) { print(); old_irq14_handler(); }
void interrupt new_irq15_handler(void) { print(); old_irq15_handler(); }
//-----------------------------------------------------------------------------

void init_new_handlers(void) {
  // System timer
  old_irq0_handler = getvect(0x08);
  setvect(MASTER_BASE_VECTOR, new_irq0_handler);

  // Keyboard controller
  old_irq1_handler = getvect(0x09);
  setvect(MASTER_BASE_VECTOR + 1, new_irq1_handler);

  // Cascaded signals from IRQs 8-15
  // (any devices configured to use IRQ2 will actually be using IRQ9)
  old_irq2_handler = getvect(0x0A);
  setvect(MASTER_BASE_VECTOR + 2, new_irq2_handler);

  // Serial port controller for serial port 2
  // (shared with serial port 4, if present)
  old_irq3_handler = getvect(0x0B);
  setvect(MASTER_BASE_VECTOR + 3, new_irq3_handler);

  // Serial port controller for serial port 1
  // (shared with serial port 3, if present)
  old_irq4_handler = getvect(0x0C);
  setvect(MASTER_BASE_VECTOR + 4, new_irq4_handler);

  // Parallel port 2 and 3 or sound card
  old_irq5_handler = getvect(0x0D);
  setvect(MASTER_BASE_VECTOR + 5, new_irq5_handler);

  // Floppy disk controller
  old_irq6_handler = getvect(0x0E);
  setvect(MASTER_BASE_VECTOR + 6, new_irq6_handler);

  // Parallel port 1. It is used for printers or for
  // any parallel port if a printer is not present
  old_irq7_handler = getvect(0x0F);
  setvect(MASTER_BASE_VECTOR + 7, new_irq7_handler);

  // Real-time clock (RTC)
  old_irq8_handler = getvect(0x70);
  setvect(SLAVE_BASE_VECTOR, new_irq8_handler);

  // Advanced Configuration and Power Interface (ACPI) system control interrupt
  old_irq9_handler = getvect(0x71);
  setvect(SLAVE_BASE_VECTOR + 1, new_irq9_handler);

  // The interrupt is left open for the use of peripherals
  // (open interrupt/available, SCSI or NIC)
  old_irq10_handler = getvect(0x72);
  setvect(SLAVE_BASE_VECTOR + 2, new_irq10_handler);

  // The interrupt is left open for the use of peripherals
  // (open interrupt/available, SCSI or NIC)
  old_irq11_handler = getvect(0x73);
  setvect(SLAVE_BASE_VECTOR + 3, new_irq11_handler);

  // Mouse on PS/2 connector
  old_irq12_handler = getvect(0x74);
  setvect(SLAVE_BASE_VECTOR + 4, new_irq12_handler);

  // CPU coprocessor or integrated floating point unit or inter-processor
  // interrupt (use depends on OS)
  old_irq13_handler = getvect(0x75);
  setvect(SLAVE_BASE_VECTOR + 5, new_irq13_handler);

  // Primary ATA channel
  // (ATA interface usually serves hard disk drives and CD drives)
  old_irq14_handler = getvect(0x76);
  setvect(SLAVE_BASE_VECTOR + 6, new_irq14_handler);

  // Secondary ATA channel
  old_irq15_handler = getvect(0x77);
  setvect(SLAVE_BASE_VECTOR + 7, new_irq15_handler);

  // Disable interruptions
  _disable();

  // Initialize master
  outp(0x20, 0x11);               // ICW1 
  outp(0x21, MASTER_BASE_VECTOR); // ICW2
  outp(0x21, 0x04);               // ICW3
  outp(0x21, 0x01);               // ICW4

  // Initialize slave
  outp(0xA0, 0x11);              // ICW1 
  outp(0xA1, SLAVE_BASE_VECTOR); // ICW2
  outp(0xA1, 0x02);              // ICW3
  outp(0xA1, 0x01);              // ICW4

  // Enable interruptions
  _enable();
}

int main(void) {
  unsigned far* fp;

  init_new_handlers();
  system("cls");

  puts("                   -  MASK");
  puts("                   -  REQUEST");
  puts("                   -  SERVICE");
  puts(" MASTER   SLAVE");

  // Make resident program
  FP_SEG(fp) = _psp;
  FP_OFF(fp) = 0x2c;
  _dos_freemem(*fp);
  _dos_keep(0, (_DS - _CS) + (_SP / 16) + 1);
  return EXIT_SUCCESS;
}
