#pragma once

#include "TRootApp.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TrainHistoryLog.h"


///
class TensorBoard {
protected:
public:
	virtual ~TensorBoard(){}
	virtual void Visualize(const TrainHistoryLog &train_history_log) = 0;
};

class TensorBoardRoot : public TensorBoard{
protected:
	TRootApp * RootApp;
	TCanvas * Canvas;		//owned by RootApp
	std::unique_ptr<TMultiGraph>	LossMultigraph,
																AccMultigraph;
public:
	TensorBoardRoot(TRootApp * root_app) : RootApp(root_app)
	{
		Canvas = RootApp->CreateCanvas("c1", "Train progress", 200, 10, 1600, 800);
		Canvas->Divide(2);
		Canvas->Draw();
	}
	virtual ~TensorBoardRoot() override
	{
	}
	virtual void Visualize(const TrainHistoryLog &train_history_log) override
	{
		LossMultigraph = std::make_unique<TMultiGraph>();
		AccMultigraph = std::make_unique<TMultiGraph>();

		//ownned by multigraphs
		TGraph * TrainLossGraph = new TGraph(train_history_log.size());
		TGraph * TrainAccGraph = new TGraph(train_history_log.size());
		TGraph * ValLossGraph = new TGraph(train_history_log.size());
		TGraph * ValAccGraph = new TGraph(train_history_log.size());


		for (size_t i = 0; i < train_history_log.size(); i++)
		{
			TrainLossGraph->SetPoint(i, train_history_log[i].Epoch, train_history_log[i].TrainLoss);
			TrainAccGraph->SetPoint(i, train_history_log[i].Epoch, train_history_log[i].TrainAcc);
			ValLossGraph->SetPoint(i, train_history_log[i].Epoch, train_history_log[i].ValLoss);
			ValAccGraph->SetPoint(i, train_history_log[i].Epoch, train_history_log[i].ValAcc);
		}

		auto draw_graphs = [](TVirtualPad * pad, TGraph * gr1, TGraph * gr2, TMultiGraph * mg, const char * xtitle, const char * ytitle)
		{
			pad->Clear();
			pad->SetLogy();
			gr1->SetLineWidth(3);
			gr1->SetLineColor(kBlue);
			gr2->SetLineWidth(3);
			gr2->SetLineColor(kRed);
			mg->Clear();
			mg->Add(gr1);
			mg->Add(gr2);
			mg->Draw("AL");

			//mg->SetTitle(obj_title);
			mg->GetXaxis()->SetLabelFont(22);
			mg->GetXaxis()->SetTitleFont(22);
			mg->GetXaxis()->SetTickLength(0.02f);
			mg->GetYaxis()->SetLabelFont(22);
			mg->GetYaxis()->SetTitleFont(22);
			mg->GetYaxis()->SetTickLength(0.02f);
			mg->GetXaxis()->SetLabelSize(0.04f);
			mg->GetXaxis()->SetLabelOffset(0.01f);
			mg->GetXaxis()->SetTitleSize(0.04f);
			mg->GetXaxis()->SetTitleOffset(1.1f);
			mg->GetYaxis()->SetLabelSize(0.04f);
			mg->GetYaxis()->SetLabelOffset(0.01f);
			mg->GetYaxis()->SetTitleSize(0.04f);
			mg->GetYaxis()->SetTitleOffset(1);
			mg->GetXaxis()->SetTitle(xtitle);
			mg->GetYaxis()->SetTitle(ytitle);
		};

		draw_graphs(Canvas->cd(1), TrainLossGraph, ValLossGraph, LossMultigraph.get(), "Epochs", "Loss");
		//Canvas->cd(1)->Update();

		draw_graphs(Canvas->cd(2), TrainAccGraph, ValAccGraph, AccMultigraph.get(), "Epochs", "Accuracy");
		//Canvas->cd(2)->Update();

		Canvas->Update();
		Canvas->Draw();
		gSystem->ProcessEvents();
	}
};
