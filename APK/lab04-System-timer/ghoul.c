#include <stdio.h>
#include <dos.h>

#define d2 73u
#define c3 130u
#define D3 156u
#define f3 175u
#define a3 220u
#define A3 233u
#define d4 294u
#define D4 311u
#define f4 349u
#define g4 392u
#define a4 440u
#define A4 466u
#define c5 523u
#define d5 587u
#define D5 622u
#define f5 698u
#define g5 784u
#define a5 880u
#define A5 932u
#define C6 1109u

#define duration 150u
#define indent   150u

#define NOTES_AMOUNT 72u

unsigned notes[NOTES_AMOUNT][3] = {
    {A5, duration, indent},
    {C6, duration * 3, indent},
    {A5, duration * 3, indent},
    {a5, duration, indent},
    {g5, duration * 3, indent},
    {C6, duration * 3, indent},
    {A5, duration * 3, indent},
    {a5, duration, indent},

    {g5, duration * 3, indent},
    {g5, duration, indent},
    {f5, duration * 5, indent},
    {f5, duration, indent},
    {D5, duration * 3, indent},
    {f5, duration, indent},
    {d5, duration * 3, indent * 13},

    {d4, duration, indent},
    {d4, duration * 3, indent},
    {d4, duration, indent},
    {d4, duration * 3, indent},
    {d5, duration, indent},

    {d5, duration * 3, indent},
    {D3, duration * 3, indent},
    {D3, duration, indent},
    {D3, duration * 3, indent},
    {D3, duration * 3, indent},
    {A4, duration, indent},
    {a4, duration * 3, indent},
    {a4, duration, indent},

    {a4, duration * 3, indent},
    {A4, duration, indent},
    {A4, duration * 3, indent},
    {D3, duration * 3, indent},
    {D3, duration, indent},
    {D3, duration * 3, indent},
    {D3, duration * 3, indent},
    {A4, duration, indent},

    {c5, duration * 3, indent},
    {A4, duration * 3, indent},
    {a4, duration, indent},
    {g4, duration * 3, indent},
    {c5, duration * 3, indent},
    {A4, duration * 3, indent},
    {a4, duration * 3, indent},

    {g4, duration * 3, indent},
    {g4, duration, indent},
    {f4, duration * 5, indent},
    {f4, duration, indent},
    {D4, duration * 3, indent},
    {f4, duration, indent},
    {d4, duration, indent},
    {c3, duration, indent},
    {A3, duration * 2, indent},

    {a3, duration * 2, indent},
    {f3, duration, indent},
    {d2, duration * 3, indent},
    {d4, duration, indent},
    {d4, duration * 3, indent},
    {d4, duration, indent},
    {d4, duration * 3, indent},
    {d5, duration, indent},
    {d5, duration * 3, indent},

    {c3, duration, indent},
    {A3, duration * 3, indent},
    {a3, duration * 3, indent},
    {f3, duration, indent},
    {d4, duration * 3, indent},
    {A5, duration, indent},
    {a5, duration * 3, indent},
    {a5, duration, indent},
    {a5, duration * 3, indent},

    {A5, duration, indent},
    {A5, duration * 3, indent},
};

void state_words(void) {
  unsigned channel, state;

  // Port 40h (channel 0, system clock interruption)
  // Port 41h (channel 1, memory regeneration)
  // Port 42h (channel 2, speaker sound)
  int ports[] = {0x40, 0x41, 0x42};

  // 11 - RBC (always 11)
  // 1 - not remember CE
  // 0 - read chanel state
  // 001, 010, 100 - chanel
  // 0 - always 0
  //                11 1 0 001 0, 11 1 0 010 0, 11 1 0 100 0
  int  control_word[] = {226, 228, 232};

  // Almost the same as control register (in set_frequency)
  // 6 - check is timer ready to read
  // 7 - OUT: check out line state
  char state_word[]   = "76000000";

  int  i;

  printf("Status word: \n");
  for (channel = 0; channel < 3; channel++) {
    // Select channel (CLC commands)
    outp(0x43, control_word[channel]);
    // Read state
    state = inp(ports[channel]);

    // Convert state into binary
    for (i = 7; i >= 0; i--) {
      state_word[i] = (char) ((state % 2) + '0');
      state /= 2;
    }
    printf("Channel %d: %s\n", channel, state_word);
  }
}

void set_frequency(unsigned divider) {
  unsigned long kd = 1193180 / divider;

  // 10 11 011 0:
  // 10 - chanel
  // 11 - read/write low, then high byte
  // 011 - meander
  // 0 - bin
  outp(0x43, 0xB6);
  // The smallest byte of the frequency divider
  outp(0x42, kd % 256);
  kd /= 256;
  // The highest byte of the frequency divider
  outp(0x42, kd);
}

void play_music(void) {
  int i;
  for (i = 0; i < NOTES_AMOUNT; i++) {
    set_frequency(notes[i][0]);
    // Turn on speaker using first 2 bits:
    // 0 - turn on/off chanel 2 in sys timer
    // 1 - turn on/off dynamic
    outp(0x61, inp(0x61) | 0x03);
    delay(notes[i][1]);
    // Turn off speaker
    outp(0x61, inp(0x61) & 0xFC);
    delay(notes[i][2]);
  }
}

int main(void) {
  state_words();
  play_music();
  state_words();
  return 0;
}
