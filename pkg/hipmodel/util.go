// Copyright (c) 2020, Stephen Polcyn. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// util.go
// Methods for manipulating datasets and error calculations

package hipmodel

import (
	"log"
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

func convert_slice_to_int(old_slice []uint32) []int {
	int_slice := make([]int, len(old_slice))
	for idx, val := range old_slice {
		int_slice[idx] = int(val)
	}

	return int_slice
}

func convert_slice_to_float64(old_slice []float32) []float64 {
	float64_slice := make([]float64, len(old_slice))
	for idx, val := range old_slice {
		float64_slice[idx] = float64(val)
	}

	return float64_slice
}

func convert_slice_to_uint32(old_slice []int) []uint32 {
	uint32_slice := make([]uint32, len(old_slice))
	for idx, val := range old_slice {
		uint32_slice[idx] = uint32(val)
	}

	return uint32_slice
}
