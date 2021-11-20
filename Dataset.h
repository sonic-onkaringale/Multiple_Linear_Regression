//
// Created by Onkar Ingale on 12-11-2021.
//

#ifndef DATASET_H
#define DATASET_H

#include "rapidcsv.h"
#include <initializer_list>
#include <vector>


namespace dataset
{
    class Dataset
    {
    private:
        std::string filepath;
        std::vector<std::vector<long double>> x_train;
        std::vector<long double> y_train;
        std::vector<std::string> x_header;
        std::string y_header;

        friend class linearreg::LinearReg;


    public:
        //Constructor
        explicit Dataset(const std::string &filepath);

        //Functions
        template<typename... Args>
        void X_train(Args... args);

        void Y_train(const std::string &y);

        //Info
        std::string shape();

        void head();

    };

    Dataset::Dataset(const std::string &filepath)
    {
        this->filepath = filepath;
    }

    template<typename... Args>
    void Dataset::X_train(Args... args)
    {
        rapidcsv::Document doc(filepath);

        x_header = {args...};
        x_train.resize(x_header.size());
        for (int i = 0; i < x_train.size(); ++i)
        {
            x_train[i] = doc.GetColumn<long double>(x_header[i]);
        }
    }

    void Dataset::Y_train(const std::string &y)
    {
        rapidcsv::Document doc(filepath);
        y_header = y;
        y_train = doc.GetColumn<long double>(y_header);
    }

    void Dataset::head()
    {
        std::cout << "\n";
        for (int i = 0; i < x_header.size(); ++i)
        {
            std::cout << " " << x_header[i];
        }
        std::cout << " " << y_header << std::endl;

    }

    std::string Dataset::shape()
    {
        return (std::to_string(x_train[0].size()) + " x " + std::to_string(x_header.size() + 1));
    }

}
#endif //DATASET_H
