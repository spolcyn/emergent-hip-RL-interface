// Copyright (c) 2020, Stephen Polcyn. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip_utils.go
// Utility methods for interacting with the hippocampus model

package hipmodel

import (
	"math"

	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"

	"github.com/goki/gi/gi"
)

// Test Slice Equality
func SlicesAreEqual(s1, s2 []int) bool {

	if len(s1) != len(s2) {
		return false
	}

	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			return false
		}
	}

	return true
}

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

// Finds the most similar vocabulary pattern to a provided activation set within a vocabulary
func FindMostSimilarVocabPattern(pattern *etensor.Float32, vocab patgen.Vocab) *etensor.Float32 {

	var distance float32 = math.MaxFloat32
	var mostSimilar *etensor.Float32

	// iterate through all vocab items, storing the one with the lowest distance from our pattern
	for _, vItem := range vocab {
		tmpDistance := CalculateDistance(pattern, vItem)

		if tmpDistance < distance {
			distance = tmpDistance
			mostSimilar = vItem
		}
	}

	return mostSimilar.Clone().(*etensor.Float32)

}

// Given a candidate and a reference, outputs a distance score between them
// Currently uses Hamming Distance
// Tensors are float-types, so can't easily use XOR as one could on a bit array
// On our size tensors, shouldn't matter -- if tensors get too big, also easily parallelizable
func CalculateDistance(t1, t2 *etensor.Float32) float32 {

	// would be better to throw an error here
	if !SlicesAreEqual(t1.ShapeObj().Shp, t2.ShapeObj().Shp) {
		panic("tensors don't have the same shape!")
	}

	var distance float32 = 0.0

	for i := 0; i < t1.ShapeObj().Len(); i++ {
		if math.Abs(math.Round(float64(t1.Value1D(i)))) != math.Abs(math.Round(float64(t2.Value1D(i)))) {
			distance++
		}
	}

	return distance
}

// Calculates which vocabulary pattern the ECOut is most similar to,
// then compares that pattern to the ECIn to see if it is correct or not
// We return the distance between the patterns for downstream functions
// to make decisions about what it means
func (ss *Sim) CalculateError(ecin *leabra.Layer, ecout *leabra.Layer) {

	// get activations
	outPattern := etensor.NewFloat32(ecout.Shape().Shp, nil, nil)
	ecout.UnitValsTensor(outPattern, "ActM") // get minus-phase activation

	DPrintf("Outpattern:\n\n%v", outPattern)

	var distance float32 = math.MaxFloat32
	var mostSimilar *etensor.Float32

	// find closest pattern in the training dataset to the output pattern
	for i := 0; i < ss.TrainAB.NumRows(); i++ {

		tmpPattern := ss.TrainAB.ColByName("Input").SubSpace([]int{i}).(*etensor.Float32)
		var tmpDistance float32 = CalculateDistance(outPattern, tmpPattern)

		if tmpDistance < distance {
			distance = tmpDistance
			mostSimilar = tmpPattern
		}
	}

	DPrintf("Distance: %v | ClosestPattern: %v", distance, mostSimilar)

	// set the error pattern
	dimensions := make([]uint32, len(mostSimilar.Shape.Shp))
	for idx, val := range mostSimilar.Shape.Shp {
		dimensions[idx] = uint32(val)
	}
	closest_tensor := Tensor{
		Version:    1,
		Dimensions: dimensions,
		Data:       convert_slice_to_float64(mostSimilar.Values)}

	name_error := &NameError{Version: 1, Distance: distance, ClosestPattern: &closest_tensor}
	ss.NameErrorResult = name_error
}

// Updates the testenv with a pattern so that the index 0 of the testenv will return the test pattern
// for use with testing a very specific, arbitrary pattern
func (ss *Sim) UpdateTestEnvWithTestPatterns(corrPattern, targPattern *etensor.Float32) {

	DPrintf("Test Tensor: \n%v\nTarget: \n%v\n", corrPattern, targPattern)

	// create columns to put patterns in as rows
	corrColWrap := etensor.NewFloat32(append([]int{1}, corrPattern.Shape.Shp...), nil, []string{"row"})
	corrColWrap.SubSpace([]int{0}).CopyFrom(corrPattern)
	targColWrap := etensor.NewFloat32(append([]int{1}, targPattern.Shape.Shp...), nil, []string{"row"})
	targColWrap.SubSpace([]int{0}).CopyFrom(targPattern)

	rowNames := etensor.NewString([]int{1}, nil, []string{"row"})
	rowNames.Set1D(0, "test-pattern")

	// create new etable with pattern
	table := etable.NewTable("TestPattern")
	table.SetNumRows(1)
	table.AddCol(rowNames, "Name")
	table.AddCol(corrColWrap, "Input")
	table.AddCol(targColWrap, "ECout")

	DPrintf("TestAB Table BEFORE: \n%v\n\n", ss.TestAB)

	// set test data
	ss.TestAB = table

	DPrintf("TestAB Table AFTER: \n%v\n\n", ss.TestAB)

	// update test environment
	ss.ConfigEnv()

	DPrintf("Test Env: \n%v\n\n", ss.TestEnv)
}
