// Copyright (c) 2020, Stephen Polcyn. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip_utils.go
// Utility methods for interacting with the hippocampus model

package hipmodel

import (
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"

	"github.com/goki/gi/gi"
)

// Opens a pattern that is stored in a CSV file using the Comma separator
func (ss *Sim) OpenPatComma(dt *etable.Table, fname, name, desc string) error {
	err := dt.OpenCSV(gi.FileName(fname), etable.Comma)
	if err != nil {
		return err
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)

	return nil
}

// Calculates which vocabulary pattern the ECOut is most similar to,
// then compares that pattern to the ECIn to see if it is correct or not
// We return the distance between the patterns for downstream functions
// to make decisions about what it means
func (ss *Sim) get_final_ecout_activations(ecout *leabra.Layer) {

	dimensions := convert_slice_to_uint32(ecout.Shp.Shp)
	out_values := make([]float32, len(ecout.Neurons))
	ecout.UnitVals(&out_values, "ActM") // get minus-phase activation

	DPrintf("Out values:\n\n%v", out_values)

	output_tensor := Tensor{
		Version:    1,
		Dimensions: dimensions,
		Data:       convert_slice_to_float64(out_values)}

	name_error := &NameError{Version: 2, OutputPattern: &output_tensor}
	ss.NameErrorResult = name_error
}

// Updates the testenv with a pattern so that the index 0 of the testenv will return the test pattern
// for use with testing a very specific, arbitrary pattern
func (ss *Sim) UpdateTestEnvWithTestPattern(corrPattern *etensor.Float32) {
	DPrintf("Corrupted Pattern: \n%v\n", corrPattern)

	// create columns to put patterns in as rows
	pattern_col := etensor.NewFloat32(append([]int{1}, corrPattern.Shape.Shp...), nil, []string{"row"})
	pattern_col.SubSpace([]int{0}).CopyFrom(corrPattern)

	rowNames := etensor.NewString([]int{1}, nil, []string{"row"})
	rowNames.Set1D(0, "test-pattern")

	// create new etable with pattern
	table := etable.NewTable("TestPattern")
	table.SetNumRows(1)
	table.AddCol(rowNames, "Name")
	table.AddCol(pattern_col, "Input")
	table.AddCol(pattern_col, "ECout")

	DPrintf("TestAB Table BEFORE: \n%v\n\n", ss.TestAB)

	// set test data
	ss.TestAB = table

	DPrintf("TestAB Table AFTER: \n%v\n\n", ss.TestAB)

	// update test environment
	ss.ConfigEnv()

	DPrintf("Test Env: \n%v\n\n", ss.TestEnv)
}
