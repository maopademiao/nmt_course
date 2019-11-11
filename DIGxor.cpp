#include "DIGxor.h"
#include "../../tensor/function/FHeader.h"
#include "../../network/XNet.h"
#include "../../tensor/function/Softmax.h"



namespace digxor
{
	/* base parameter */
	float learningRate = 0.3F;		//learning rate
	int nEpoch = 100;				//max training epoches
	float minmax = 0.01F;			// range[-p,p] for parameter initialization

	void Init(DIGxorModel &model);
	void Train(float (*trainDataX)[2], float *trainDataY, int dataSize, DIGxorModel &model);
	void InitGrad(DIGxorModel &model, DIGxorModel &grad);
	void Forword(XTensor &input, DIGxorModel &model, DIGxorNet &net);
	void MSELoss(XTensor &output, XTensor &goal, XTensor &loss);
	void Backward(XTensor &input, XTensor &goal, DIGxorModel &model, DIGxorModel &grad, DIGxorNet &net);
	void Update(DIGxorModel &model, DIGxorModel &grad, float learningRate);
	/*void CleanGrad(DIGxorModel &grad);
	void Test(float *testData, int testDataSize, DIGxorModel &model);*/

	int DIGxorMain(int argc, const char ** argv)
	{
		DIGxorModel model;
		model.h_size1 = 3;
		model.h_size2 = 3;
		const int dataSize1 = 4;
		const int dataSize2 = 2;
		const int testDataSize = 3;
		model.devID = -1;
		Init(model);

		/* train Data */
		float trainDataX[dataSize1][dataSize2] = { 0,0,1,0,0,1,1,1 };
		float trainDataY[dataSize1] = { 0,1,1,0 };

		float testDataX[testDataSize][dataSize2] = { 0,1,1,0,0,0 };
		Train(trainDataX, trainDataY, dataSize1, model);

		return 0;
	}

	void Init(DIGxorModel &model)
	{
		//printf("init-----\n");
		InitTensor2D(&model.weight1, 2, model.h_size1, X_FLOAT, model.devID);
		InitTensor2D(&model.weight2, model.h_size1, model.h_size2, X_FLOAT, model.devID);
		InitTensor2D(&model.weight3, model.h_size2, 1, X_FLOAT, model.devID);
		InitTensor2D(&model.b1, 1, model.h_size1, X_FLOAT, model.devID);
		InitTensor2D(&model.b2, 1, model.h_size2, X_FLOAT, model.devID);
		model.weight1.SetDataRand(-minmax, minmax);
		//model.weight1.Dump(stderr);
		model.weight2.SetDataRand(-minmax, minmax);
		model.weight3.SetDataRand(-minmax, minmax);
		model.b1.SetZeroAll();
		model.b2.SetZeroAll();
		printf("Init model finish!\n");
	}

	void InitGrad(DIGxorModel &model, DIGxorModel &grad)
	{
		InitTensor(&grad.weight1, &model.weight1);
		//printf("grad==========\n");
		//grad.weight1.Dump(stderr);
		InitTensor(&grad.weight2, &model.weight2);
		InitTensor(&grad.weight3, &model.weight3);
		InitTensor(&grad.b1, &model.b1);
		InitTensor(&grad.b2, &model.b2);
		grad.h_size1 = model.h_size1;
		grad.h_size2 = model.h_size2;
		grad.devID = model.devID;
	}

	void Train(float(*trainDataX)[2], float *trainDataY, int dataSize, DIGxorModel &model)
	{
		printf("prepare data for train\n");
		/* prepare for train */
		TensorList inputList;
		TensorList goalList;
		for (int i = 0; i < dataSize; ++i)
		{
			
			XTensor* inputData = NewTensor2D(1, 2, X_FLOAT, model.devID);
			inputData->Set2D(trainDataX[i][0], 0, 0);
			inputData->Set2D(trainDataX[i][1], 0, 1);
			inputList.Add(inputData);
			//inputData->Dump(stderr);
			
			XTensor* goalData = NewTensor2D(1, 1, X_FLOAT, model.devID);
			goalData->Set2D(trainDataY[i], 0, 0);
			goalList.Add(goalData);
		}

		printf("start train\n");
		DIGxorNet net;
		DIGxorModel grad;
		InitGrad(model, grad);
		XNet autoDiffer;
		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
		{
			printf("epoch %d\n", epochIndex);
			float totalLoss = 0;
			if ((epochIndex + 1) % 50 == 0)
				learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)
			{
				printf("%d\n",inputList.count);
				//break;
				XTensor *input = inputList.GetItem(i);
				XTensor *goal = goalList.GetItem(i);
				Forword(*input, model, net);
				printf("\n\n");
				model.weight1.Dump(stderr);
				printf("\n\n");
				//net.output.Dump(stderr);
				XTensor loss;
				//MSELoss(net.output, *goal, loss);
				//loss.Dump(stderr);
				XTensor lossTensor;
				lossTensor = CrossEntropy(net.output, goal);
				//totalLoss += loss.Get1D(0);

				autoDiffer.Backward(lossTensor);
				//Backward(XTensor &input, XTensor &goal, DIGxorModel &model, DIGxorModel &grad, DIGxorNet &net);
				Update(model, grad, learningRate);
			}
			break;
		}
	}

	void Forword(XTensor &input, DIGxorModel &model, DIGxorNet &net)
	{
		net.hidden_state1 = MatrixMul(input, model.weight1);
		net.hidden_state2 = net.hidden_state1 + model.b1;
		net.hidden_state3 = Softmax(net.hidden_state2,1);
		net.hidden_state4 = MatrixMul(net.hidden_state3, model.weight2);
		net.hidden_state5 = net.hidden_state4 + model.b2;
		net.hidden_state6 = Softmax(net.hidden_state5,1);
		net.output = Softmax(MatrixMul(net.hidden_state6, model.weight3),1);
	}

	void MSELoss(XTensor &output, XTensor &goal, XTensor &loss)
	{
		XTensor tmp = output - goal;
		loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
	}

	void MSELossBackword(XTensor &output, XTensor &goal, XTensor &grad)
	{
		XTensor tmp = output - goal;
		grad = tmp * 2;
	}

	void Backward(XTensor &input, XTensor &goal, DIGxorModel &model, DIGxorModel &grad, DIGxorNet &net)
	{

	 }

	void Update(DIGxorModel &model, DIGxorModel &grad, float learningRate)
	{
		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
		model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
		model.weight3 = Sum(model.weight3, grad.weight3, -learningRate);
		model.b1 = Sum(model.b1, grad.b1, -learningRate);
		model.b2 = Sum(model.b2, grad.b2, -learningRate);
	}
}