// Copyright (c) 2020, Stephen Polcyn. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// server.go
// Implements the REST API server for the hippocampus model

package hipmodel

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"sync"

	"github.com/golang/protobuf/proto"
	"github.com/labstack/echo"
)

/* The hippocampus API server, managing an API server and a hippocampus model */
type HipServer struct {
	es  *echo.Echo // echo server
	sim *Sim       // the hippocampus simulation
	mu  *sync.Mutex
}

/* Creates and runs the hippocampus API server */
func (hs *HipServer) Init(address string, sim *Sim) {
	// setup properties
	hs.sim = sim
	hs.es = echo.New()
	hs.mu = &sync.Mutex{}

	// setup server
	hs.setupRoutes()

	// setup logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// start server
	go func() { hs.es.Logger.Fatal(hs.es.Start(address)) }()
}

/*** API-Related Data Types ***/

type TrainRequest struct {
	MaxRuns int `json:"maxruns" query:"maxruns" form:"maxruns"` // max runs to train for
	MaxEpcs int `json:"maxepcs" query:"maxepcs" form:"maxepcs"` // max epochs to train for
}

/*** END API-Related Data Types ***/

/* Setup the API endpoints */
func (hs *HipServer) setupRoutes() {

	get_protobuf_binary := func(req *http.Request) []byte {
		data, err := ioutil.ReadAll(req.Body)
		if err != nil {
			log.Fatalf("Unable to read message from request: %v", err)
		}

		return data
	}

	// update training data to new dataset
	hs.es.PUT("/dataset/train/update", func(c echo.Context) error {

		var err error

		// read in request
		update := new(DatasetUpdate)
		data := get_protobuf_binary(c.Request())
		err = proto.Unmarshal(data, update)
		if err != nil {
			log.Fatalf("Unable to unmarshal update from request: %v", err)
		}

		DPrintf("Dataset update: %v", update)

		// lock and update model
		hs.mu.Lock()
		err = hs.sim.RestUpdateTrainingData(update)
		hs.mu.Unlock()

		// send response
		if err == nil {
			return c.String(http.StatusOK, fmt.Sprintf("Training dataset updated from source: %v", update.Source))
		} else {
			return c.String(http.StatusBadRequest, err.Error())
		}

	})

	// test a pattern
	hs.es.POST("/model/testpattern", func(c echo.Context) error {

		// read in request
		test_request := new(TestItem)
		data := get_protobuf_binary(c.Request())
		err := proto.Unmarshal(data, test_request)
		if err != nil {
			log.Fatalf("Unable to unmarshal update from request: %v", err)
		}

		DPrintf("Test item data: %v", test_request)

		// lock and test the item
		hs.mu.Lock()
		nameError, err := hs.sim.RestTestPattern(test_request)
		hs.mu.Unlock()

		// send response
		if err == nil {
			name_error_binary, err2 := proto.Marshal(nameError)

			if err2 != nil {
				DPrintf("proto marshaling error, %v", err2.Error())
			}
			return c.HTMLBlob(http.StatusOK, name_error_binary)

		} else {
			return c.String(http.StatusOK, err.Error())
		}
	})

	// start training the model
	hs.es.POST("/model/train", func(c echo.Context) error {

		// read in request
		tr := new(TrainRequest)

		if err := c.Bind(tr); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		// lock and start training
		hs.mu.Lock()
		str, err := hs.sim.RestStartTraining(tr)
		hs.mu.Unlock()

		// send response
		if err == nil {
			return c.String(http.StatusOK, str)
		} else {
			return c.String(http.StatusPreconditionFailed, err.Error())
		}
	})
}
