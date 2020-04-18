#!/usr/local/bin/pyleabra

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py 
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# to run this python version of the demo:
# * install gopy, currently in fork at https://github.com/goki/gopy
#   e.g., 'go get github.com/goki/gopy -u ./...' and then cd to that package
#   and do 'go install'
# * go to the python directory in this emergent repository, read README.md there, and 
#   type 'make' -- if that works, then type make install (may need sudo)
# * cd back here, and run 'pyemergent' which was installed into /usr/local/bin
# * then type 'import ra25' and this should run
# * you'll need various standard packages such as pandas, numpy, matplotlib, etc

# labra25ra runs a simple random-associator 5x5 = 25 four-layer leabra network

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, epygiv

import importlib as il  #il.reload(ra25) -- doesn't seem to work for reasons unknown
import io, sys, getopt
# import numpy as np
# import matplotlib
# matplotlib.use('SVG')
# import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'  # essential for not rendering fonts as paths

# note: pandas, xarray or pytorch TensorDataSet can be used for input / output
# patterns and recording of "log" data for plotting.  However, the etable.Table
# has better GUI and API support, and handles tensor columns directly unlike
# pandas.  Support for easy migration between these is forthcoming.
# import pandas as pd

# this will become Sim later.. 
TheSim = 1

# use this for e.g., etable.Column construction args where nil would be passed
nilInts = go.Slice_int()

# use this for e.g., etable.Column construction args where nil would be passed
nilStrs = go.Slice_string()

# LogPrec is precision for saving float values in logs
LogPrec = 4

# note: we cannot use methods for callbacks from Go -- must be separate functions
# so below are all the callbacks from the GUI toolbar actions

def InitCB(recv, send, sig, data):
    TheSim.Init()
    TheSim.ClassView.Update()
    TheSim.vp.SetNeedsFullRender()

def TrainCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.Train()

def StopCB(recv, send, sig, data):
    TheSim.Stop()

def StepTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TrainTrial()
        TheSim.IsRunning = False
        TheSim.ClassView.Update()
        TheSim.vp.SetNeedsFullRender()

def StepEpochCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.TrainEpoch()

def StepRunCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.TrainRun()

def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestTrial()
        TheSim.IsRunning = False
        TheSim.ClassView.Update()
        TheSim.vp.SetNeedsFullRender()

def TestItemCB2(recv, send, sig, data):
    win = gi.Window(handle=recv)
    vp = win.WinViewport2D()
    dlg = gi.Dialog(handle=send)
    if sig != gi.DialogAccepted:
        return
    val = gi.StringPromptDialogValue(dlg)
    idxs = TheSim.TestEnv.Table.RowsByString("Name", val, True, True) # contains, ignoreCase
    if len(idxs) == 0:
        gi.PromptDialog(vp, gi.DlgOpts(Title="Name Not Found", Prompt="No patterns found containing: " + val), True, False, go.nil, go.nil)
    else:
        if not TheSim.IsRunning:
            TheSim.IsRunning = True
            print("testing index: %s" % idxs[0])
            TheSim.TestItem(idxs[0])
            TheSim.IsRunning = False
            vp.SetNeedsFullRender()

def TestItemCB(recv, send, sig, data):
    win = gi.Window(handle=recv)
    gi.StringPromptDialog(win.WinViewport2D(), "", "Test Item",
        gi.DlgOpts(Title="Test Item", Prompt="Enter the Name of a given input pattern to test (case insensitive, contains given string."), win, TestItemCB2)

def TestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunTestAll()

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")

def FilterSSE(et, row):
    return etable.Table(handle=et).CellFloat("SSE", row) > 0 # include error trials    

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

#####################################################    
#     Sim

class Sim(object):
    """
    Sim encapsulates the entire simulation model, and we define all the
    functionality as methods on this struct.  This structure keeps all relevant
    state information organized and available without having to pass everything around
    as arguments to methods, and provides the core GUI interface (note the view tags
    for the fields which provide hints to how things should be displayed).
    """
    def __init__(self):
        ss.Net = leabra.Network()
        self.TrainAB = etable.Table()
        self.TrainAC = etable.Table()
        self.TestAB = etable.Table()
        self.TestAC = etable.Table()
        self.TestLure = etable.Table()
        self.TrnTrlLog = etable.Table()
        self.TrnEpcLog = etable.Table()
        self.TstEpcLog = etable.Table()
        self.TstTrlLog = etable.Table()
        self.TstCycLog = etable.Table()
        self.RunLog = etable.Table()
        self.RunStats = etable.Table()
        ss.Params = ParamSets
        # ss.Params = SavedParamsSets
        ss.RndSeed = 2
        ss.ViewOn = true
        ss.TrainUpdt = leabra.AlphaCycle
        ss.TestUpdt = leabra.Cycle
        ss.TestInterval = 1
        ss.LogSetParams = false
        ss.MemThr = 0.34
        ss.LayStatNms = []string{"ECin", "DG", "CA3", "CA1"}
        ss.TstNms = []string{"AB", "AC", "Lure"}
        ss.TstStatNms = []string{"Mem", "TrgOnWasOff", "TrgOffWasOn"}

        # statistics
        self.TrlSSE     = 0.0
        self.TrlAvgSSE  = 0.0
        self.TrlCosDiff = 0.0
        self.EpcSSE     = 0.0
        self.EpcAvgSSE  = 0.0
        self.EpcPctErr  = 0.0
        self.EpcPctCor  = 0.0
        self.EpcCosDiff = 0.0
        self.FirstZero  = -1

        # internal state - view:"-"
        self.SumSSE     = 0.0
        self.SumAvgSSE  = 0.0
        self.SumCosDiff = 0.0
        self.CntErr     = 0.0
        self.Win        = 0
        self.vp         = 0
        self.ToolBar    = 0
        self.NetView    = 0
        self.TrnEpcPlot = 0
        self.TstEpcPlot = 0
        self.TstTrlPlot = 0
        self.TstCycPlot = 0
        self.RunPlot    = 0
        self.TrnEpcFile = 0
        self.RunFile    = 0
        self.InputValsTsr = 0
        self.OutputValsTsr = 0
        self.SaveWts    = False
        self.NoGui        = False
        self.LogSetParams = False # True
        self.IsRunning    = False
        self.StopNow    = False
        self.RndSeed    = 0

        # statistics
        self.TrlSSE     = 0.0
        self.TrlAvgSSE  = 0.0
        self.TrlCosDiff = 0.0
        self.EpcSSE     = 0.0
        self.EpcAvgSSE  = 0.0
        self.EpcPctErr  = 0.0
        self.EpcPctCor  = 0.0
        self.EpcCosDiff = 0.0
        self.FirstZero  = -1

        # internal state - view:"-"
        self.SumSSE     = 0.0
        self.SumAvgSSE  = 0.0
        self.SumCosDiff = 0.0
        self.CntErr     = 0.0
        self.Win        = 0
        self.vp         = 0
        self.ToolBar    = 0
        self.NetView    = 0
        self.TrnEpcPlot = 0
        self.TstEpcPlot = 0
        self.TstTrlPlot = 0
        self.TstCycPlot = 0
        self.RunPlot    = 0
        self.TrnEpcFile = 0
        self.RunFile    = 0
        self.InputValsTsr = 0
        self.OutputValsTsr = 0
        self.SaveWts    = False
        self.NoGui        = False
        self.LogSetParams = False # True
        self.IsRunning    = False
        self.StopNow    = False
        self.RndSeed    = 0

        # ClassView tags for controlling display of fields
        self.Tags = {
            'TestNm': 'inactive:"+"',
            'Mem': 'inactive:"+"',
            'TrgOnWasOffAll': 'inactive:"+"',
            'TrgOnWasOffCmp': 'inactive:"+"',
            'TrgOffWasOn': 'inactive:"+"',
            'TrlSSE': 'inactive:"+"',
            'TrlAvgSSE': 'inactive:"+"',
            'TrlCosDiff': 'inactive:"+"',

            'EpcSSE': 'inactive:"+"',
            'EpcAvgSSE': 'inactive:"+"',
            'EpcPctErr': 'inactive:"+"',
            'EpcPctCor': 'inactive:"+"',
            'EpcCosDiff': 'inactive:"+"',
            'FirstZero': 'inactive:"+"',
            'NZero': 'inactive:"+"',

            # internal state - view:"-"
            'SumSSE': 'view:"-"',
            'SumAvgSSE': 'view:"-"',
            'SumCosDiff': 'view:"-"',
            'CntErr': 'view:"-"',
            'Win': 'view:"-"',
            'NetView': 'view:"-"',
            'ToolBar': 'view:"-"',
            'TrnTrlPlot': 'view:"-"',
            'TrnEpcPlot': 'view:"-"',
            'TstEpcPlot': 'view:"-"',
            'TstTrlPlot': 'view:"-"',
            'TstCycPlot': 'view:"-"',
            'RunPlot': 'view:"-"',
            'TrnEpcFile': 'view:"-"',
            'RunFile': 'view:"-"',
            'TmpVals': 'view:"-"',
            'LayStatNms': 'view:"-"',
            'TstNms': 'view:"-"',
            'TstStatNms': 'view:"-"',
            'SaveWts': 'view:"-"',
            'NoGui': 'view:"-"',
            'LogSetParams': 'view:"-"',
            'IsRunning': 'view:"-"',
            'StopNow': 'view:"-"',
            'NeedsNewRun': 'view:"-"',
            'RndSeed': 'view:"-"',

            'ClassView': 'view:"-"',
            'Tags': 'view:"-"',
        }

    def InitParams(self):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        self.Params.OpenJSON("hip_std.params")

        # todo: the following expression SHOULD produce the same results but it ends up
        # adding the items in a random order relative to what is shown here -- each time
        # the order is different.  very strange
        # pars = params.Set(Name="Base", Desc="these are the best params", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="Prjn", Desc="norm and momentum on works better, but wt bal is not better for smaller nets",
        #                 Params=params.Params({
        #                     "Prjn.Learn.Norm.On":     "true",
        #                     "Prjn.Learn.Momentum.On": "true",
        #                     "Prjn.Learn.WtBal.On":    "false",
        #                 }).handle),
        #             params.Sel(Sel="Layer", Desc="using default 1.8 inhib for all of network -- can explore",
        #                 Params=params.Params({
        #                     "Layer.Inhib.Layer.Gi": "1.8",
        #                 }).handle),
        #             params.Sel(Sel="#Output", Desc="output definitely needs lower inhib -- true for smaller layers in general",
        #                 Params=params.Params({
        #                     "Layer.Inhib.Layer.Gi": "1.4",
        #                 }).handle),
        #             params.Sel(Sel=".Back", Desc="top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
        #                 Params=params.Params({
        #                     "Prjn.WtScale.Rel": "0.2",
        #                 }).handle),
        #             }).handle,
        #         "Sim": params.Sheet({
        #             params.Sel(Sel="Sim", Desc="best params always finish in this time",
        #                 Params=params.Params({
        #                     "Sim.MaxEpcs": "50",
        #                 }).handle),
        #             }).handle,
        #     }).handle),
        # params.Set(Name="DefaultInhib", Desc="output uses default inhib instead of lower", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="#Output", Desc="go back to default",
        #                 Params=params.Params({
        #                     "Layer.Inhib.Layer.Gi": "1.8",
        #                    }).handle),
        #                 }).handle,
        #         "Sim": params.Sheet({
        #             params.Sel(Sel="Sim", Desc="takes longer -- generally doesn't finish..",
        #                 Params=params.Params({
        #                     "Sim.MaxEpcs": "100",
        #                }).handle),
        #             }).handle,
        #      }).handle),
        # params.Set(Name="NoMomentum", Desc="no momentum or normalization", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="Prjn", Desc="no norm or momentum",
        #                 Params=params.Params({
        #                     "Prjn.Learn.Norm.On":     "false",
        #                     "Prjn.Learn.Momentum.On": "false",
        #                 }).handle),
        #             }).handle,
        #         }).handle),
        # params.Set(Name="WtBalOn", Desc="try with weight bal on", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="Prjn", Desc="weight bal on",
        #                Params=params.Params({
        #                    "Prjn.Learn.WtBal.On": "true",
        #                }).handle),
        #            }).handle,
        #        }).handle),
        # })

    ######################################
    #   Configs

    def Config(self):
        """Config configures all the elements using the standard functions"""
        self.InitParams()
        self.OpenPats()
        self.ConfigPats()
        self.ConfigEnv()
        self.ConfigNet(self.Net)
        self.ConfigTrnTrlLog(self.TrnTrlLog)
        self.ConfigTrnEpcLog(self.TrnEpcLog)
        self.ConfigTstEpcLog(self.TstEpcLog)
        self.ConfigTstTrlLog(self.TstTrlLog)
        self.ConfigTstCycLog(self.TstCycLog)
        self.ConfigRunLog(self.RunLog)

    def ConfigEnv(self): 
        if self.MaxRuns == 0: # allow user override
            self.MaxRuns = 10
        if self.MaxEpcs == 0: # allow user override
            self.MaxEpcs = 50
        
        self.TrainEnv.Nm = "TrainEnv"
        self.TrainEnv.Dsc = "training params and state"
        self.TrainEnv.Table = etable.NewIdxView(self.Pats)
        self.TrainEnv.Validate()
        self.TrainEnv.Run.Max = self.MaxRuns # note: we are not setting epoch max -- do that manually
        
        self.TestEnv.Nm = "TestEnv"
        self.TestEnv.Dsc = "testing params and state"
        self.TestEnv.Table = etable.NewIdxView(self.Pats)
        self.TestEnv.Sequential = True
        self.TestEnv.Validate()
        
        # note: to create a train / test split of pats, do this:
        # all = etable.NewIdxView(self.Pats)
        # splits = split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
        # self.TrainEnv.Table = splits.Splits[0]
        # self.TestEnv.Table = splits.Splits[1]
        
        self.TrainEnv.Init(0)
        self.TestEnv.Init(0)


    # SetEnv select which set of patterns to train on: AB or AC
    def SetEnv(self, trainAC): 
        if trainAC: 
                self.TrainEnv.Table = etable.NewIdxView(self.TrainAC)
         else:
                ss.TrainEnv.Table = etable.NewIdxView(self.TrainAB)
                
        self.TrainEnv.Init(0)

    def ConfigNet(self, net):
	net.InitName(net, "Hip")
	inLay = net.AddLayer4D("Input", 6, 2, 3, 4, emer.Input)
	ecin = net.AddLayer4D("ECin", 6, 2, 3, 4, emer.Hidden)
	ecout = net.AddLayer4D("ECout", 6, 2, 3, 4, emer.Target) # clamped in plus phase
	ca1 = net.AddLayer4D("CA1", 6, 2, 4, 10, emer.Hidden)
	dg = net.AddLayer2D("DG", 25, 25, emer.Hidden)
	ca3 = net.AddLayer2D("CA3", 30, 10, emer.Hidden)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	ecin.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="Input", YAlign=relpos.Front, Space: 2))
	ecout.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="ECin", YAlign=relpos.Front, Space=2))
	dg.SetRelPos(relpos.Rel(Rel=relpos.Above, Other="Input", YAlign=relpos.Front, XAlign=relpos.Left, Space=0))
	ca3.SetRelPos(relpos.Rel(Rel=relpos.Above, Other="DG", YAlign=relpos.Front, XAlign=relpos.Left, Space=0))
	ca1.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="CA3", YAlign=relpos.Front, Space=2})

	net.ConnectLayers(inLay, ecin, prjn.NewOneToOne(), emer.Forward)
	net.ConnectLayers(ecout, ecin, prjn.NewOneToOne(), emer.Back)

	# EC <-> CA1 encoder pathways
	pj := net.ConnectLayersPrjn(ecin, ca1, prjn.NewPoolOneToOne(), emer.Forward, hip.EcCa1Prjn())
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ca1, ecout, prjn.NewPoolOneToOne(), emer.Forward, hip.EcCa1Prjn())
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ecout, ca1, prjn.NewPoolOneToOne(), emer.Back, hip.EcCa1Prjn())
	pj.SetClass("EcCa1Prjn")

	# Perforant pathway
	ppath = prjn.NewUnifRnd()
	ppath.PCon = 0.25

	pj = net.ConnectLayersPrjn(ecin, dg, ppath, emer.Forward, hip.CHLPrjn())
	pj.SetClass("HippoCHL")
	pj = net.ConnectLayersPrjn(ecin, ca3, ppath, emer.Forward, hip.CHLPrjn())
	pj.SetClass("HippoCHL")

	# Mossy fibers
	mossy := prjn.NewUnifRnd()
	mossy.PCon = 0.05
	pj = net.ConnectLayersPrjn(dg, ca3, mossy, emer.Forward, hip.CHLPrjn()) # no learning
	pj.SetClass("HippoCHL")

	# Schafer collaterals
	pj = net.ConnectLayersPrjn(ca3, ca3, prjn.NewFull(), emer.Lateral, hip.CHLPrjn())
	pj.SetClass("HippoCHL")
	pj = net.ConnectLayersPrjn(ca3, ca1, prjn.NewFull(), emer.Forward, hip.CHLPrjn())
	pj.SetClass("HippoCHL")

	# using 3 threads :)
	dg.SetThread(1)
	ca3.SetThread(2)
	ca1.SetThread(3)

	# note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	# outLay.SetType(emer.Compare)
	# that would mean that the output layer doesn't reflect target values in plus phase
	# and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) # only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

########################################
### 	    Init, utils

# Init restarts the run, and initializes everything, including network weights
# and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() # re-config env just in case a different set of patterns was
	# selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) # all sheets
	ss.NewRun()
	ss.UpdateView(true)
}
