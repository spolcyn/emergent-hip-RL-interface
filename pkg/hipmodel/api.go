// api.go
// implements the methods interacting between the model and the server

package hipmodel

import (
	"errors"
	"fmt"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"log"
	"time"
)

// Update the data used for training the model
func (ss *Sim) RestUpdateTrainingData(update *DatasetUpdate) error {

	//log.Printf("update: %v\n", update)

	// update dataset from file
	if update.Source == "file" {
		log.Printf("updating training data with file: %v\n", update.Filename)
		err := ss.OpenPatComma(ss.TrainAB, update.Filename, "TrainAB", "AB Training Patterns")
		return err
	} else if update.Source == "body" {
		//log.Printf("pattern len: %v\n", len(update.Patterns))
		//log.Printf("body: \n\n%v\n", update.Patterns)
		log.Printf("updating training data with %v patterns\n", len(update.Patterns))

		// parse patterns to array of etensors
		pats := make([]*etensor.Float32, len(update.Patterns))
		for i, jsonPat := range update.Patterns {
			//log.Printf("parsing:\n\n%v\n\nwith shape: %v\n", v, update.Shape)
			pats[i] = ParseTensorFromJSON(update.Shape, jsonPat)
		}

		// Create training etable
		trainpats := etable.NewTable("Training Patterns")
		trainpats.SetMetaData("desc", "Training data from API")
		trainpats.SetNumRows(len(pats))

		// get pattern shape and compute column shape
		patShape := pats[0].Shape.Shp
		colShape := append([]int{len(pats)}, patShape...)
		//log.Printf("Patshape: %v || Colshape: %v", patShape, colShape)

		// setup columns for filling
		rowNames := etensor.NewString([]int{len(pats)}, nil, []string{"row"})
		col := etensor.NewFloat32(colShape, nil, []string{"row"})

		// copy array of patterns into a single etensor column
		for i, tsr := range pats {
			//log.Printf("i: %v || tsr: \n%v\n", i, tsr)
			rowNames.Set1D(i, fmt.Sprintf("trn-%v", i))
			col.SubSpace([]int{i}).CopyFrom(tsr)
		}

		//log.Printf("Column: \n\n%v\n\n", col)

		// add the patterns to the table
		trainpats.AddCol(rowNames, "Name")
		trainpats.AddCol(col, "Input")
		trainpats.AddCol(col, "ECout")

		// Set training etable in model
		//log.Printf("\n\nTrainAB BEFORE\n\n%v\n", ss.TrainAB)
		ss.TrainAB = trainpats
		//log.Printf("\n\nTrainAB AFTER\n\n%v\n", ss.TrainAB)

		//log.Printf("parsed patterns:\n\n%v\n", pats)

		return nil
	} else {
		return errors.New("Invalid update method")
	}
}

/* Update the data used for comparing output patterns to for error determination */
func (ss *Sim) RestUpdateInputPatternData(update *DatasetUpdate) error {
	log.Printf("updating input patterns data with file: %v\n", update.Filename)

	// as it turns out, TestAB is where the IdxView for the TestItem comes from, so we need to update that
	// NOTE: for now, we're still opening InputPatterns as well, but long-term that shouldn't happen
	ss.OpenPatComma(ss.InputPatterns, update.Filename, "InputPatterns", "Input Patterns")

	return nil
}

/* Test an item */
func (ss *Sim) RestTestPattern(tr *TestRequest) (*NameError, error) {
	if ss.IsRunning {
		return nil, errors.New("Model is already running, couldn't test item yet")
	}

	ss.IsRunning = true
	log.Printf("testing pattern: %v\n", tr.Pattern)

	// setup the environment as we want it
	tsr := ParseTensorFromJSON(tr.Shape, tr.Pattern)
	ss.UpdateEnvWithTestPattern(tsr)

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

	// periodically check to see if the training is done, then return.
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
