// Copyright (c) 2020, Stephen Polcyn. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// util.go
// Methods for manipulating datasets and error calculations

package hipmodel

import (
	"log"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/emer/etable/etensor"
)

// Debugging - set to >0 to enable DPrintf statements
// Approach taken from code provided by COS418 (f'19) at Princeton University
const Debug = 0

func DPrintf(format string, a ...interface{}) (n int, err error) {

	if Debug > 0 {
		log.Printf(format, a...)
	}
	return
}

// checks if the given rune is in the map
func isin(m map[rune]struct{}, r rune) bool {
	_, ok := m[r]
	return ok
}

// parses a tensor from a JSON format
// not sure what the standard JSON format for an n-d array is yet though, will probably use whatever
// the standard dump format for a numpy array is
func ParseTensorFromJSON(shapeJSON, patternJSON string) (pattern *etensor.Float32) {

	DPrintf("shapeJSON: %v\n\npatternJSON: %v\n\n", shapeJSON, patternJSON)

	// parse the shape of the tensor and create it
	re := regexp.MustCompile(`\d+`)
	shapeStrings := re.FindAllString(shapeJSON, -1)
	shape := make([]int, len(shapeStrings))

	for i, v := range shapeStrings {
		if s, err := strconv.Atoi(v); err == nil {
			shape[i] = s
		}
	}

	DPrintf("shapestrings: %v\n", shapeStrings)
	DPrintf("shape: %v\n", shape)
	pattern = etensor.NewFloat32(shape, nil, nil)

	// create an n-d coordinate tracking where we're inserting data
	// think of it like a writing head that puts the next piece of data in a particular place, and we consistently move it
	// so that we keep putting data in the correct coordinate
	writeCoor := make([]int, len(shape))

	// parse the data into the etensor
	reader := strings.NewReader(patternJSON)
	currentD := -1 // current dimension being worked on -- start at -1 so first [ brings us to 0th dimension

	// setup maps to enable multiple valid separator characters
	increaseD := make(map[rune]struct{})
	increaseD['['] = struct{}{}
	increaseD['('] = struct{}{}
	decreaseD := make(map[rune]struct{})
	decreaseD[']'] = struct{}{}
	decreaseD[')'] = struct{}{}

	for reader.Len() > 0 {
		if r, _, err := reader.ReadRune(); err == nil {
			switch {
			case isin(increaseD, r):
				currentD += 1
			case isin(decreaseD, r):
				currentD -= 1
			case unicode.IsDigit(r):
				//				fmt.Printf("writeCoor: %v\n", writeCoor)
				pattern.Set(writeCoor, float32(r-'0')) // copy to tensor
				writeCoor[currentD] += 1               // increment writing head

				// only accepting binary input currently
				if float32(r-'0') != 1.0 && float32(r-'0') != 0.0 {
					DPrintf("expected 0 or 1, got %v\n", float32(r-'0'))
					panic("Number not 0 or 1 in input")
				}
			}
		}
	}

	DPrintf("Pattern: %v\n", pattern)

	return pattern
}
