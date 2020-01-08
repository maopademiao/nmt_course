#pragma once
#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace digxor2
{
	struct DIGxorModel
	{
		XTensor weight1;
		XTensor b;
		int h_size;
		int devID;
	};

	struct DIGxorNet
	{
		/* before bias ===> mul w1 */
		XTensor hidden_state1;
		/* add bias */
		XTensor hidden_state2;
		/* add activate fuction */
		XTensor hidden_state3;
		/* output */
		XTensor output;
	};
	int DIGxorMain3(int argc, const char ** argv);
}
