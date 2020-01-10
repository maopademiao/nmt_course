#include "../homework/digxor2.h"
#include "../../tensor/function/FHeader.h"
#include "../../network/XNet.h"
#include "../../tensor/function/Sigmoid.h"
#include "../../tensor/loss/CrossEntropy.h"
#include "../../tensor/function/Softmax.h"

namespace digxor2
{
	/* base parameter */
	float learningRate = 0.005F;		//learning rate
	int nEpoch = 1500;				//max training epoches
	float minmax = 0.99F;			// range[-p,p] for parameter initialization

	void Init(DIGxorModel &model);
	void Train(int(*trainDataX)[6], int(*trainDataY)[8], int dataSize, int dataSize2, int dataSize3, DIGxorModel &model);
	void InitGrad(DIGxorModel &model, DIGxorModel &grad);
	void Forword(XTensor &input, DIGxorModel &model, DIGxorNet &net);
	void Backward(XTensor &input, XTensor &goal, DIGxorModel &model, DIGxorModel &grad, DIGxorNet &net);
	void Update(DIGxorModel &model, DIGxorModel &grad, float learningRate);
	void CleanGrad(DIGxorModel &grad);
	void Test(int(*testData)[6], int testDataSize, DIGxorModel &model);
	int DIGxorMain3(int argc, const char ** argv)
	{
		DIGxorModel model;
		model.h_size = 8;
		const int dataSize1 = 64;
		const int dataSize2 = 2;
		const int dataSize2_2 = 6;
		const int dataSize3 = 8;
		model.devID = -1;
		Init(model);

		/* train Data */
		float trainDataX[dataSize1][dataSize2] = { 0,0,0,1,0,2,0,3,0,4,0,5,0,6,0,7,
			1,0,1,1,1,2,1,3,1,4,1,5,1,6,1,7,
			2,0,2,1,2,2,2,3,2,4,2,5,2,6,2,7,
			3,0,3,1,3,2,3,3,3,4,3,5,3,6,3,7,
			4,0,4,1,4,2,4,3,4,4,4,5,4,6,4,7,
			5,0,5,1,5,2,5,3,5,4,5,5,5,6,5,7,
			6,0,6,1,6,2,6,3,6,4,6,5,6,6,6,7,
			7,0,7,1,7,2,7,3,7,4,7,5,7,6,7,7 };
		int trainDataX2[dataSize1][dataSize2_2] = { 0 };
		int trainDataY[dataSize1][dataSize3] = { 0 };

		for (int i = 0; i < dataSize1; i++)
		{
			int tmpx0 = (int)trainDataX[i][0];
			int tmpx1 = (int)trainDataX[i][1];
			int tmp = ((int)trainDataX[i][0]) ^ ((int)trainDataX[i][1]);
			trainDataY[i][tmp] = 1.0;
			//printf("%d-%d ",tmp,trainDataY[i][tmp]);
			if (tmpx0 == 0) { trainDataX2[i][0] = 0; trainDataX2[i][1] = 0; trainDataX2[i][2] = 0; }
			else if (tmpx0 == 1) { trainDataX2[i][0] = 0; trainDataX2[i][1] = 0; trainDataX2[i][2] = 1; }
			else if (tmpx0 == 2) { trainDataX2[i][0] = 0; trainDataX2[i][1] = 1; trainDataX2[i][2] = 0; }
			else if (tmpx0 == 3) { trainDataX2[i][0] = 0; trainDataX2[i][1] = 1; trainDataX2[i][2] = 1; }
			else if (tmpx0 == 4) { trainDataX2[i][0] = 1; trainDataX2[i][1] = 0; trainDataX2[i][2] = 0; }
			else if (tmpx0 == 5) { trainDataX2[i][0] = 1; trainDataX2[i][1] = 0; trainDataX2[i][2] = 1; }
			else if (tmpx0 == 6) { trainDataX2[i][0] = 1; trainDataX2[i][1] = 1; trainDataX2[i][2] = 0; }
			else if (tmpx0 == 7) { trainDataX2[i][0] = 1; trainDataX2[i][1] = 1; trainDataX2[i][2] = 1; }

			if (tmpx1 == 0) { trainDataX2[i][3] = 0; trainDataX2[i][4] = 0; trainDataX2[i][5] = 0; }
			else if (tmpx1 == 1) { trainDataX2[i][3] = 0; trainDataX2[i][4] = 0; trainDataX2[i][5] = 1; }
			else if (tmpx1 == 2) { trainDataX2[i][3] = 0; trainDataX2[i][4] = 1; trainDataX2[i][5] = 0; }
			else if (tmpx1 == 3) { trainDataX2[i][3] = 0; trainDataX2[i][4] = 1; trainDataX2[i][5] = 1; }
			else if (tmpx1 == 4) { trainDataX2[i][3] = 1; trainDataX2[i][4] = 0; trainDataX2[i][5] = 0; }
			else if (tmpx1 == 5) { trainDataX2[i][3] = 1; trainDataX2[i][4] = 0; trainDataX2[i][5] = 1; }
			else if (tmpx1 == 6) { trainDataX2[i][3] = 1; trainDataX2[i][4] = 1; trainDataX2[i][5] = 0; }
			else if (tmpx1 == 7) { trainDataX2[i][3] = 1; trainDataX2[i][4] = 1; trainDataX2[i][5] = 1; }

			//break;
		}
		/*
		for (int i = 0; i < 64; i++)
		{
		for (int j = 0; j < 6; j++)
		{
		printf("%d ",trainDataX2[i][j]);
		}
		printf("\n");
		}*/
		const int testDataSize = 1;
		int testDataX[testDataSize][dataSize2_2] = { 0,0,0,0,0,0 };
		Train(trainDataX2, trainDataY, dataSize1, dataSize2_2, dataSize3, model);
		//Test(testDataX, testDataSize, model);
		//Test(trainDataX2, dataSize1, model);
		return 0;
	}

	void Init(DIGxorModel &model)
	{
		//printf("init-----\n");
		InitTensor2D(&model.weight1, 6, model.h_size, X_FLOAT, model.devID);
		//初始化W为6*8的tensor
		InitTensor2D(&model.b, 1, model.h_size, X_FLOAT, model.devID);
		//初始化b为1*8的tensor
		model.weight1.SetDataRand(-minmax, minmax);
		//model.weight1.Dump(stderr);
		model.b.SetZeroAll();
		printf("Init model finish!\n");
	}

	void InitGrad(DIGxorModel &model, DIGxorModel &grad)
	{
		InitTensor(&grad.weight1, &model.weight1);
		InitTensor(&grad.b, &model.b);
		grad.h_size = model.h_size;
		grad.devID = model.devID;
	}

	void Train(int(*trainDataX)[6], int(*trainDataY)[8], int dataSize, int dataSize2, int dataSize3, DIGxorModel &model)
	{
		/*  trainDataX为dataSize*dataSize2的输入数组
		trainDataY为dataSize*dataSize3的输出数组
		*/
		printf("prepare data for train\n");
		/* prepare for train */
		TensorList inputList;
		TensorList goalList;
		for (int i = 0; i < dataSize; ++i)
		{
			XTensor* inputData = NewTensor2D(1, dataSize2, X_FLOAT, model.devID);//1*6
			for (int tmpi = 0; tmpi<dataSize2; tmpi++)
				inputData->Set2D(trainDataX[i][tmpi], 0, tmpi);
			inputList.Add(inputData);
			//inputData->Dump(stderr);
			XTensor* goalData = NewTensor2D(1, dataSize3, X_FLOAT, model.devID);//1*8
			for (int tmpi = 0; tmpi<dataSize3; tmpi++)
				goalData->Set2D(trainDataY[i][tmpi], 0, tmpi);
			goalList.Add(goalData);
		}

		printf("start train\n");
		DIGxorNet net;
		DIGxorModel grad;
		InitGrad(model, grad);

		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
		{
			//printf("epoch %d\n", epochIndex);
			float totalLoss = 0;
			//if ((epochIndex + 1) % 50 == 0)
			//	learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)
			{
				XTensor *input = inputList.GetItem(i);
				//input->Dump(stderr);
				XTensor *goal = goalList.GetItem(i);
				//goal->Dump(stderr);
				Forword(*input, model, net);

				//printf("forward ok\n");
				//net.output.Dump(stderr);
				XTensor loss;
				//printf("begin crossentropy\n");
				loss = CrossEntropy(net.output, goal, 0);
				//loss.Dump(stderr);
				totalLoss += loss.Get1D(0);
				Backward(*input, *goal, model, grad, net);
				//printf("%f\n", totalLoss);
				Update(model, grad, learningRate);
				CleanGrad(grad);
				//break;
			}
			//Test(trainDataX, dataSize, model);
			if (epochIndex % 100 == 0)
			{
				printf("epoch %d\n", epochIndex);
				printf("loss:%f\n", totalLoss / inputList.count);
				Test(trainDataX, dataSize, model);
			}

			//break;
		}
	}

	void Forword(XTensor &input, DIGxorModel &model, DIGxorNet &net)
	{
		/*  h1 = w1x
		h2 = w1x+b
		h3 = sigmoid(h2)
		output = softmax(h3)
		*/
		net.hidden_state1 = MatrixMul(input, model.weight1);
		net.hidden_state2 = net.hidden_state1 + model.b;
		net.hidden_state3 = Sigmoid(net.hidden_state2);
		net.output = Softmax(net.hidden_state3, 1);
		//printf("\nforward\n");
		//net.output.Dump(stderr);
	}

	void Backward(XTensor &input, XTensor &goal, DIGxorModel &model, DIGxorModel &grad, DIGxorNet &net)
	{
		//printf("\nbegin backward\n");
		XTensor lossGrad;
		XTensor &out = net.output;
		XTensor &h3 = net.hidden_state3;
		XTensor dedy(&out);
		XTensor dedx(&h3);
		XTensor &dedb = grad.b;
		XTensor &dedw1 = grad.weight1;
		//_CrossEntropyBackward(&dedy, &net.output, &goal, &dedw1);
		/* dedy = dE/dout, dedx = dE/dh3 */
		_SoftmaxBackward(&goal, &out, &h3, &dedy, &dedx, NULL, 1, CROSSENTROPY);
		/* dedb = dedx * dh3 / dh2 */
		//_HardTanHBackward(&net.hidden_state3, &net.hidden_state2, &dedx, &dedb);
		_SigmoidBackward(&net.hidden_state3, &net.hidden_state2, &dedx, &dedb);
		dedw1 = MatrixMul(input, X_TRANS, dedb, X_NOTRANS);
	}

	void Update(DIGxorModel &model, DIGxorModel &grad, float learningRate)
	{
		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
		model.b = Sum(model.b, grad.b, -learningRate);
	}
	void CleanGrad(DIGxorModel &grad)
	{
		grad.b.SetZeroAll();
		grad.weight1.SetZeroAll();
	}
	void Test(int(*testDataX)[6], int testDataSize, DIGxorModel &model)
	{
		//printf("\nTest\n");
		DIGxorNet net;
		TensorList inputList;
		TensorList resultlist;
		int t1[64], t2[64];
		for (int i = 0; i < 64; ++i)
		{
			//printf("\ni:%d\n", i);
			XTensor*  inputData = NewTensor2D(1, 6, X_FLOAT, model.devID);
			for (int tmp1 = 0; tmp1<6; tmp1++)
				inputData->Set2D(testDataX[i][tmp1], 0, tmp1);
			t1[i] = testDataX[i][0] * 4 + testDataX[i][1] * 2 + testDataX[i][2] * 1;
			t2[i] = testDataX[i][3] * 4 + testDataX[i][4] * 2 + testDataX[i][5] * 1;
			inputList.Add(inputData);
		}
		int anslist[64] = {};
		int right = 0;
		for (int i = 0; i < inputList.count; ++i)
		{
			XTensor *input = inputList.GetItem(i);
			Forword(*input, model, net);
			float firstans = net.output.Get2D(0, 0);
			int pos = 0;
			for (int k = 1; k < 8; k++)
			{
				float ans = net.output.Get2D(0, k);
				if (ans > firstans)
				{
					firstans = ans;
					pos = k;
				}
			}
			if (pos == (t1[i] ^ t2[i])) right++;
		}
		printf("right:%d\n", right);
	}
}
