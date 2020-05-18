// api.go
// Author: Stephen Polcyn
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

	DPrintf("Update: %v\n", update)
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
		DPrintf("pattern len: %v\n", len(update.Patterns))
		DPrintf("body: \n\n%v\n", update.Patterns)
		DPrintf("updating training data with %v patterns\n", len(update.Patterns))

		// parse patterns to array of etensors
		pats := make([]*etensor.Float32, len(update.Patterns))
		for i, jsonPat := range update.Patterns {
			DPrintf("parsing:\n\n%v\n\nwith shape: %v\n", jsonPat, update.Shape)
			pats[i] = ParseTensorFromJSON(update.Shape, jsonPat)
		}

		// Create training etable
		trainpats := etable.NewTable("Training Patterns")
		trainpats.SetMetaData("desc", "Training data from API")
		trainpats.SetNumRows(len(pats))

		// get pattern shape and compute column shape
		patShape := pats[0].Shape.Shp
		colShape := append([]int{len(pats)}, patShape...)
		DPrintf("Patshape: %v || Colshape: %v", patShape, colShape)

		// setup columns for filling
		rowNames := etensor.NewString([]int{len(pats)}, nil, []string{"row"})
		col := etensor.NewFloat32(colShape, nil, []string{"row"})

		// copy array of patterns into a single etensor column
		for i, tsr := range pats {
			DPrintf("i: %v || tsr: \n%v\n", i, tsr)
			rowNames.Set1D(i, fmt.Sprintf("trn-%v", i))
			col.SubSpace([]int{i}).CopyFrom(tsr)
		}

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

		DPrintf("parsed patterns:\n\n%v\n", pats)
	} else {
		return errors.New("Invalid update method")
	}

	return err
}

/* Test an item */
func (ss *Sim) RestTestPattern(tr *TestRequest) (*NameError, error) {
	if ss.IsRunning {
		return nil, errors.New("Model is already running, couldn't test item yet")
	}

	ss.IsRunning = true
	DPrintf("testing pattern: %v\n", tr.CorruptedPattern)

	// setup the environment as we want it
	corrTsr := ParseTensorFromJSON(tr.Shape, tr.CorruptedPattern)
	targTsr := ParseTensorFromJSON(tr.Shape, tr.TargetPattern)
	ss.UpdateTestEnvWithTestPatterns(corrTsr, targTsr)

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
