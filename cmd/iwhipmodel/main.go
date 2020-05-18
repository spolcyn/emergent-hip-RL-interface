package main

import "github.com/spolcyn/leabra/examples/iwhipmodel/pkg/hipmodel"
import "os/signal"
import "os"
import "fmt"

func main() {

	// initalize the hippocampus model server
	hipmodel.InitModel()

	// setup wait for CTRL+C interrupt so that program doesn't exit immediately on launch
	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, os.Interrupt)

	// wait for signal
	<-sigc

	fmt.Println("Got signal, exiting")
}
