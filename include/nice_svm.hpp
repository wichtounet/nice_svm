//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef NICE_SVM_HPP
#define NICE_SVM_HPP

#include <random>
#include <utility>
#include <cassert>

#include "svm.h"

namespace svm {

struct problem {
    std::size_t n_samples;
    std::size_t n_features;
    svm_problem sub;

    problem(std::size_t n_samples, std::size_t n_features) :
            n_samples(n_samples),
            n_features(n_features),
            sub({static_cast<int>(n_samples), new double[n_samples], new svm_node*[n_samples]}){
        //Init the sub vectors
        for(std::size_t s = 0; s < n_samples; ++s){
            sub.x[s] = new svm_node[n_features+1];
        }
    }

    problem(problem& rhs) = delete;

    problem(problem&& rhs) :
            n_samples(rhs.n_samples),
            n_features(rhs.n_features),
            sub({rhs.sub.l, rhs.sub.y, rhs.sub.x}){
        rhs.sub.x = nullptr;
        rhs.sub.y = nullptr;
    }

    ~problem(){
        if(sub.x){
            for(std::size_t s = 0; s < n_samples; ++s){
                delete[] sub.x[s];
            }

            delete[] sub.x;
        }

        if(sub.y){
            delete[] sub.y;
        }
    }

    svm_problem& get_problem(){
        return sub;
    }

    const svm_problem& get_problem() const {
        return sub;
    }

    double& label(std::size_t i){
        return sub.y[i];
    }

    svm_node* sample(std::size_t i){
        return sub.x[i];
    }
};

struct model {
    svm_model* sub = nullptr;

    model() = default;

    model(svm_model* sub) : sub(sub) {}

    model(model&& rhs) : sub(rhs.sub){
        rhs.sub = nullptr;
    }

    model& operator=(model&& rhs){
        sub = rhs.sub;
        rhs.sub = nullptr;

        return *this;
    }

    ~model(){
        if(sub){
            svm_free_and_destroy_model(&sub);
        }
    }

    svm_model* get_model(){
        return sub;
    }

    std::size_t classes(){
        return svm_get_nr_class(sub);
    }

    const svm_model* get_model() const {
        return sub;
    }

    operator bool(){
        return sub;
    }
};

template<typename IT1, typename IT2, typename RNG>
void parallel_shuffle(IT1 first_1, IT1 last_1, IT2 first_2, IT2 last_2, RNG&& g){
    assert(std::distance(first_1, last_1) == std::distance(first_2, last_2));

    typedef typename std::iterator_traits<IT1>::difference_type diff_t;
    typedef typename std::make_unsigned<diff_t>::type udiff_t;
    typedef typename std::uniform_int_distribution<udiff_t> distr_t;
    typedef typename distr_t::param_type param_t;

    distr_t D;
    diff_t n = last_1 - first_1;

    for (diff_t i = n-1; i > 0; --i) {
        using std::swap;
        auto new_i = D(g, param_t(0, i));
        swap(first_1[i], first_1[new_i]);
        swap(first_2[i], first_2[new_i]);
    }
}

template<typename Labels, typename Images>
problem make_problem(Labels& labels, Images& samples, std::size_t max = 0, bool shuffle = true){
    assert(labels.size() == samples.size());

    if(shuffle){
        static std::random_device rd;
        static std::mt19937_64 g(rd());

        parallel_shuffle(samples.begin(), samples.end(), labels.begin(), labels.end(), g);
    }

    if(max > 0 && max < labels.size()){
        labels.resize(max);
        samples.resize(max);
    }

    auto n_samples = labels.size();

    problem problem(n_samples, samples.front().size());

    for(std::size_t s = 0; s < n_samples; ++s){
        auto features = samples[s].size();

        problem.label(s) = labels[s];

        for(std::size_t i = 0; i < features; ++i){
            problem.sample(s)[i].index = i+1;
            problem.sample(s)[i].value = samples[s][i];
        }

        //End the vector
        problem.sample(s)[features].index = -1;
        problem.sample(s)[features].value = 0.0;
    }

    return problem;
}

inline svm_parameter default_parameters(){
    svm_parameter parameters;

    parameters.svm_type = C_SVC;
    parameters.kernel_type = RBF;
    parameters.degree = 3;
    parameters.gamma = 0;
    parameters.coef0 = 0;
    parameters.nu = 0.5;
    parameters.cache_size = 100;
    parameters.C = 1;
    parameters.eps = 1e-3;
    parameters.p = 0.1;
    parameters.shrinking = 1;
    parameters.probability = 0;
    parameters.nr_weight = 0;
    parameters.weight_label = nullptr;
    parameters.weight = nullptr;

    return parameters;
}

inline void test_model(problem& problem, model& model){
    double prob_estimates[10]; //TODO 10 is not fixed

    std::size_t correct = 0;

    for(std::size_t s = 0; s < problem.n_samples; ++s){
        auto label = svm_predict_probability(model.get_model(), problem.sample(s), prob_estimates);

        if(label == problem.label(s)){
            ++correct;
        }
    }

    std::cout << "Samples: " << problem.n_samples << std::endl;
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Accuracy: " << (100.0 * correct / problem.n_samples) << "%" << std::endl;
    std::cout << "Error: " << (100.0 - (100.0 * correct / problem.n_samples)) << "%" << std::endl;
}

inline svm_model* train(problem& problem, svm_parameter& parameters){
    std::cout << "Train SVM: " << problem.n_samples << " samples" << std::endl;

    auto model = svm_train(&problem.get_problem(), &parameters);

    std::cout << "Training done" << std::endl;

    return model;
}

inline void cross_validate(problem& problem, svm_parameter& parameters, std::size_t n_fold){
    std::cout << "Cross validation" << std::endl;

    double *target = new double[problem.n_samples];

    svm_cross_validation(&problem.get_problem(), &parameters, n_fold, target);

    std::size_t cross_correct = 0;

    for(std::size_t i = 0; i < problem.n_samples; ++i){
        if(target[i] == problem.label(i)){
            ++cross_correct;
        }
    }

    std::cout << "Cross validation Samples: " << problem.n_samples << std::endl;
    std::cout << "Cross validation Correct: " << cross_correct << std::endl;
    std::cout << "Cross validation Accuracy: " << (100.0 * cross_correct / problem.n_samples) << "%" << std::endl;
    std::cout << "Cross validation Error: " << (100.0 - (100.0 * cross_correct / problem.n_samples)) << "%" << std::endl;

    delete[] target;

    std::cout << "Cross validation done" << std::endl;
}

inline bool check(const problem& problem, const svm_parameter& parameters){
    auto error = svm_check_parameter(&problem.get_problem(), &parameters);

    if(error){
        std::cerr << "Parameters not checked: " << error << std::endl;

        return false;
    }

    return true;
}

inline model load(const std::string& file_name){
    std::cout << "Load SVM model" << std::endl;

    auto model = svm_load_model(file_name.c_str());

    if(!model){
        std::cout << "Impossible to load model" << std::endl;
    } else {
        std::cout << "SVM model loaded" << std::endl;
    }

    return {model};
}

inline bool save(const model& model, const std::string& file_name){
    return !svm_save_model(file_name.c_str(), model.get_model());
}

inline void print_null(const char* /*s*/) {}

inline void make_quiet(){
    svm_set_print_string_function(&print_null);
}

} //end of namespace svm

#endif