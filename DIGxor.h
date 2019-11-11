#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace digxor
{
	struct DIGxorModel
	{
		XTensor weight1;
		XTensor weight2;
		XTensor weight3;
		XTensor b1;
		XTensor b2;
		int h_size1;
		int h_size2;
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
		/* mul w2 */
		XTensor hidden_state4;
		/* add bias */
		XTensor hidden_state5;
		/* add activate fuction */
		XTensor hidden_state6;
		/* output */
		XTensor output;
	};
	int DIGxorMain(int argc, const char ** argv);
}