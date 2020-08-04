// Copyright (c) 2020, Stephen Polcyn. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// api.go
// Implements the methods interacting between the model and the server

package hipmodel

import (
	"errors"
	"fmt"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"time"
)

/* Update the data used for training/testing the model and re-init the network */
func (ss *Sim) RestUpdateTrainingData(update *DatasetUpdate) error {
	dataset := update.Dataset
	DPrintf("Update dataset: %v\n", dataset)

	var err error = nil

	// update dataset from file
	if update.Source == "file" {
		DPrintf("updating training data with file: %v\n", update.Filename)

		err = ss.OpenPatComma(ss.TrainAB, update.Filename, "Training Patterns", "Training Patterns")
		if err != nil {
			return err
		}
		err = ss.OpenPatComma(ss.TestAB, update.Filename, "Testing Patterns", "Same as Training Patterns")

	} else if update.Source == "body" {
		DPrintf("protobuf dimensions: %v", dataset.Dimensions)
		DPrintf("protobuf patterns: %v", dataset.Data)

		// parse patterns to array of etensors
		patterns := etensor.NewFloat32(convert_slice_to_int(dataset.Dimensions), nil, nil)
		patterns.SetFloats(dataset.Data)

		// Create training etable
		trainpats := etable.NewTable("Training Patterns")
		trainpats.SetMetaData("desc", "Training data from API")
		num_patterns := patterns.Shape.Shp[0]
		trainpats.SetNumRows(num_patterns)

		// setup columns for filling
		rowNames := etensor.NewString([]int{num_patterns}, nil, []string{"row"})
		col := etensor.NewFloat32(patterns.Shape.Shp, nil, []string{"row"})
		col.CopyFrom(patterns)

		DPrintf("Column: \n\n%v\n\n", col)

		// add the pattern columns to the table
		trainpats.AddCol(rowNames, "Name")
		trainpats.AddCol(col, "Input")
		trainpats.AddCol(col, "ECout")

		// Set training etable in model
		DPrintf("\n\nTrainAB BEFORE\n\n%v\n", ss.TrainAB)
		ss.TrainAB = trainpats
		ss.TestAB = trainpats
		DPrintf("\n\nTrainAB AFTER\n\n%v\n", ss.TrainAB)

		// re-init model
		ss.Init()

		DPrintf("\n\nTrain Env AFTER: \n\n%v\n", ss.TrainEnv)
		DPrintf("\n\nTest Env AFTER: \n\n%v\n", ss.TestEnv)
	} else {
		return errors.New("Invalid update method")
	}

	return err
}

/* Test an item */
func (ss *Sim) RestTestPattern(tr *TestItem) (*NameError, error) {
	if ss.IsRunning {
		return nil, errors.New("Model is already running, couldn't test item yet")
	}

	ss.IsRunning = true
	DPrintf("testing pattern: %v\n", tr.CorruptedPattern)

	// setup the environment as we want it
	corrupted_tensor := etensor.NewFloat32(convert_slice_to_int(tr.CorruptedPattern.Dimensions), nil, nil)
	corrupted_tensor.SetFloats(tr.CorruptedPattern.Data)

	target_tensor := etensor.NewFloat32(convert_slice_to_int(tr.TargetPattern.Dimensions), nil, nil)
	target_tensor.SetFloats(tr.TargetPattern.Data)

	ss.UpdateTestEnvWithTestPatterns(corrupted_tensor, target_tensor)

	ss.TestItem(0) // always use 0, that's where we'll put the item
	ss.IsRunning = false

	return ss.NameErrorResult, nil
}

/* Start the model's training process */
func (ss *Sim) RestStartTraining(tr *TrainRequest) (string, error) {

	if ss.IsRunning {
		return "", errors.New("Training is already running")
	}

	ss.MaxRuns = tr.MaxRuns
	ss.MaxEpcs = tr.MaxEpcs

	// re-init model to clear previous weights and reset training parameters
	ss.Init()

	// start the training in a goroutine with a channel for completion
	doneCh := make(chan bool)
	ss.IsRunning = true
	go func() { ss.Train(); doneCh <- true }()

	// wait for training to complete, recording time
	start := time.Now()
	<-doneCh
	end := time.Now()

	elapsed := end.Sub(start)

	return fmt.Sprintf("Training completed in %v seconds. Max Runs: %v, Max Epochs: %v\n", elapsed.Seconds(), ss.MaxRuns, ss.MaxEpcs), nil
}

/* Check on the model's training status */
func (ss *Sim) GetTrainingStatus() string {
	if ss.IsRunning {
		return "Training"
	} else {
		return "Not Training"
	}
}
