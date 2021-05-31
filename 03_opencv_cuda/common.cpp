//============================================================================
// Name        : common.cpp
// Author      : Pritesh
// Version     :
// Copyright   : Your copyright notice
// Description : common/general functions
//============================================================================

#include <common.h>

bool Common::is_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
} //source : https://stackoverflow.com/a/4654718