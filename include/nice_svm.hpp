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

    problem() : n_samples(0), n_features(0), sub({0, nullptr, nullptr}) {
        //Nothing to init
    }

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

    problem& operator=(problem&& rhs){
        n_samples = rhs.n_samples;
        n_features = rhs.n_features;
        sub.l = rhs.sub.l;
        sub.x = rhs.sub.x;
        sub.y = rhs.sub.y;

        rhs.sub.x = nullptr;
        rhs.sub.y = nullptr;

        return *this;
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
    ((void)last_2); //Ensure no warning is issued for last_2 (used only in debug mode)

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

template<typename LIterator, typename IIterator>
problem make_problem(LIterator lfirst, LIterator llast, IIterator ifirst, IIterator ilast){
    assert(std::distance(lfirst, llast) == std::distance(ifirst, ilast));
    ((void)ilast); //Ensure no warning is issued for ilast (used only in debug mode)

    auto n_samples = std::distance(lfirst, llast);

    problem problem(n_samples, ifirst->size());

    std::size_t s = 0;

    while(lfirst != llast){
        problem.label(s) = *lfirst;

        auto features = ifirst->size();

        for(std::size_t i = 0; i < features; ++i){
            problem.sample(s)[i].index = i+1;
            problem.sample(s)[i].value = (*ifirst)[i];
        }

        //End the vector
        problem.sample(s)[features].index = -1;
        problem.sample(s)[features].value = 0.0;

        ++lfirst;
        ++ifirst;
        ++s;
    }

    return problem;
}

template<typename Labels, typename Images>
problem make_problem(const Labels& labels, const Images& samples){
    assert(labels.size() == samples.size());

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

inline double predict(model& model, const svm_node* sample){
    std::vector<double> prob_estimates(model.classes());

    return svm_predict_probability(model.get_model(), sample, &prob_estimates[0]);
}

template<typename Sample>
typename std::enable_if<std::is_same<typename Sample::value_type, double>::value, double>::type predict(model& model, const Sample& sample){
    std::vector<double> prob_estimates(model.classes());

    auto features = sample.size();
    std::vector<svm_node> svm_sample(features+1);

    for(std::size_t i = 0; i < features; ++i){
        svm_sample[i].index = i+1;
        svm_sample[i].value = sample[i];
    }

    //End the vector
    svm_sample[features].index = -1;
    svm_sample[features].value = 0.0;

    return svm_predict_probability(model.get_model(), &svm_sample[0], &prob_estimates[0]);
}

inline void test_model(problem& problem, model& model){
    std::vector<double> prob_estimates(model.classes());

    std::size_t correct = 0;

    for(std::size_t s = 0; s < problem.n_samples; ++s){
        auto label = svm_predict_probability(model.get_model(), problem.sample(s), &prob_estimates[0]);

        if(label == problem.label(s)){
            ++correct;
        }
    }

    std::cout << "Samples: " << problem.n_samples << std::endl;
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Accuracy: " << (100.0 * correct / problem.n_samples) << "%" << std::endl;
    std::cout << "Error: " << (100.0 - (100.0 * correct / problem.n_samples)) << "%" << std::endl;
}

inline svm_model* train(problem& problem, const svm_parameter& parameters){
    std::cout << "Train SVM: " << problem.n_samples << " samples" << std::endl;

    auto model = svm_train(&problem.get_problem(), &parameters);

    std::cout << "Training done" << std::endl;

    return model;
}

inline double cross_validate(problem& problem, const svm_parameter& parameters, std::size_t n_fold, bool quiet = false){
    if(!quiet){
        std::cout << "Cross validation" << std::endl;
    }

    double *target = new double[problem.n_samples];

    svm_cross_validation(&problem.get_problem(), &parameters, n_fold, target);

    std::size_t cross_correct = 0;

    for(std::size_t i = 0; i < problem.n_samples; ++i){
        if(target[i] == problem.label(i)){
            ++cross_correct;
        }
    }

    if(!quiet){
        std::cout << "Cross validation Samples: " << problem.n_samples << std::endl;
        std::cout << "Cross validation Correct: " << cross_correct << std::endl;
        std::cout << "Cross validation Accuracy: " << (100.0 * cross_correct / problem.n_samples) << "%" << std::endl;
        std::cout << "Cross validation Error: " << (100.0 - (100.0 * cross_correct / problem.n_samples)) << "%" << std::endl;

        std::cout << "Cross validation done" << std::endl;
    }

    delete[] target;

    return 100.0 * cross_correct / problem.n_samples;
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

inline void rbf_grid_search(svm::problem& problem, const svm_parameter& parameters, std::size_t n_fold, const std::vector<double>& c_values, const std::vector<double>& gamma_values){
    std::cout << "Grid Search" << std::endl;

    double max_accuracy = 0.0;
    std::size_t max_C = 0;
    std::size_t max_gamma = 0;

    for(auto& C : c_values){
        for(auto& gamma : gamma_values){
            svm_parameter new_parameter = parameters;

            new_parameter.C = C;
            new_parameter.gamma = gamma;

            auto accuracy = svm::cross_validate(problem, new_parameter, n_fold, true);

            std::cout << "C=" << C << ",y=" << gamma << " -> " << accuracy << std::endl;

            if(accuracy > max_accuracy){
                max_accuracy = accuracy;
                max_C = C;
                max_gamma = gamma;
            }
        }
    }

    std::cout << "Best: C=" << max_C << ",y=" << max_gamma << " -> " << max_accuracy << std::endl;
}

enum class grid_search_type {
    LINEAR,
    EXP
};

struct rbf_grid {
    grid_search_type type = grid_search_type::EXP;

    double c_first = 2e-5;
    double c_last = 2e15;
    double c_steps = 10;

    double gamma_first = 2e-15;
    double gamma_last = 2e3;
    double gamma_steps = 10;
};

inline void rbf_grid_search_exp(svm::problem& problem, const svm_parameter& parameters, std::size_t n_fold, const rbf_grid& g = rbf_grid()){
    std::vector<double> c_values(g.c_steps);
    std::vector<double> gamma_values(g.gamma_steps);

    double c_first = g.c_first;
    double gamma_first = g.gamma_first;

    for(std::size_t i = 0; i < g.c_steps; ++i){
        c_values[i] = c_first;
        c_first *= std::pow(g.c_last / g.c_first, 1.0 / (g.c_steps - 1.0));
    }

    for(std::size_t i = 0; i < g.gamma_steps; ++i){
        gamma_values[i] = gamma_first;
        gamma_first *= std::pow(g.gamma_last / g.gamma_first, 1.0 / (g.gamma_steps - 1.0));
    }

    rbf_grid_search(problem, parameters, n_fold, c_values, gamma_values);
}

inline void rbf_grid_search_lin(svm::problem& problem, const svm_parameter& parameters, std::size_t n_fold, const rbf_grid& g = rbf_grid()){
    std::vector<double> c_values(g.c_steps);
    std::vector<double> gamma_values(g.gamma_steps);

    double c_first = g.c_first;
    double gamma_first = g.gamma_first;

    for(std::size_t i = 0; i < g.c_steps; ++i){
        c_values[i] = c_first;
        c_first += (g.c_last - g.c_first) / (g.c_steps - 1.0);
    }

    for(std::size_t i = 0; i < g.gamma_steps; ++i){
        gamma_values[i] = gamma_first;
        gamma_first += (g.gamma_last - g.gamma_first) / (g.gamma_steps - 1.0);
    }

    rbf_grid_search(problem, parameters, n_fold, c_values, gamma_values);
}

inline void rbf_grid_search(svm::problem& problem, const svm_parameter& parameters, std::size_t n_fold, const rbf_grid& g = rbf_grid()){
    switch(g.type){
        case grid_search_type::LINEAR:
            rbf_grid_search_lin(problem, parameters, n_fold, g);
        case grid_search_type::EXP:
            rbf_grid_search_exp(problem, parameters, n_fold, g);
    }
}

} //end of namespace svm

#endif