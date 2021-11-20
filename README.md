# Multiple Linear Regression in C++.

#### Installation :

1. Download the zip of this repository.

2. Include the header file 

   ```c++
   #include "LinearReg.h"
   ```

3. Enjoy :)

#### Usage : 

```cpp
#include <iostream>
#include "LinearReg.h"

using namespace dataset;
using namespace linearreg;
using namespace std;

int main()
{
    Dataset self("C:/Users/Onkar Ingale/Desktop/Drills/ML/Matrix/Startups.csv");
    self.X_train("R&D Spend","Administration","Marketing Spend");
    self.Y_train("Profit");
    LinearReg reg(self);
    cout<<"\n Intercept(Beta 0) : "<<reg.intercept()<<endl<<"\n Coefficients : "<<reg.coef();
    cout<<"\n R2 SCORE :";
    reg.r2_score();
    return 0;
}

```

#### [Class LinearReg](https://github.com/sonic-onkaringale/Multiple_Linear_Regression/blob/main/Core.h)

#### [Class Dataset](https://github.com/sonic-onkaringale/Multiple_Linear_Regression/blob/main/Dataset.h)

#### [Class Matrix](https://github.com/sonic-onkaringale/Multiple_Linear_Regression/blob/main/Matrix.h)

#### 
