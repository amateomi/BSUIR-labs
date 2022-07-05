#ifndef LAB09_INPUT__OUTPUT_AND_FILE_HANDLING_MENU_HPP
#define LAB09_INPUT__OUTPUT_AND_FILE_HANDLING_MENU_HPP

#include <iostream>
#include <fstream>
#include <utility>
#include <list>

#include "../Airport/Airport.hpp"

// File names
constexpr char text[] = "text.txt";
constexpr char textBin[] = "text-bin.txt";
constexpr char bin[] = "bin.bin";

// Erase all files
void eraseFiles(std::ofstream& target);

///////////////////////////////////////////////////////////////////////////////
// Output /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Display main menu and return range of options
std::pair<int, int> showMainMenu();

// Display menu for 1 option from main menu
std::pair<int, int> showInputMenu();

// Display menu with file choice
std::pair<int, int> showFileNames();

// Display Airport fields names
std::pair<int, int> showAirportFields();

// Write record in different files
void writeRecord(int type, std::ofstream& target, Airport& record);

// Read record form different files
void readRecord(int type, std::ifstream& source, Airport& record);

///////////////////////////////////////////////////////////////////////////////
// Input //////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Return one option form range
int getMenuOption(std::pair<int, int>& range);

// Get int from std::cin, throw std::runtime_error if fail
int readInt();

///////////////////////////////////////////////////////////////////////////////
// Menu functions /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Input record to file/copy from file to file
void inputRecords(std::ofstream& target);

// Display content form all files
void displayFiles(std::ifstream& source);

// Display records according to entered key
void searchWithKey(std::ifstream& source);

// Display records according to entered range of values
void searchInRange(std::ifstream& source);

///////////////////////////////////////////////////////////////////////////////
// File handling //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Open stream for specific file
// Use getMenuOption() as argument
void openOutputStream(int type, std::ofstream& file);

// Open stream for specific file
// Use getMenuOption() as argument
void openInputStream(int type, std::ifstream& file);

#endif //LAB09_INPUT__OUTPUT_AND_FILE_HANDLING_MENU_HPP
