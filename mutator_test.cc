// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mutator.h"

#include <memory>
#include <random>

#include "definitions.h"
#include "instruction.pb.h"
#include "algorithm.h"
#include "algorithm_test_util.h"
#include "generator.h"
#include "generator_test_util.h"
#include "mutator.pb.h"
#include "random_generator.h"
#include "test_util.h"
#include "gtest/gtest.h"

namespace automl_zero
{

    using ::std::function;    // NOLINT
    using ::std::make_shared; // NOLINT
    using ::std::mt19937;     // NOLINT
    using ::std::shared_ptr;  // NOLINT
    using ::std::vector;      // NOLINT
    using ::testing::Test;

    TEST(MutatorTest, Runs)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [ "
                "  ALTER_PARAM_MUTATION_TYPE, "
                "  RANDOMIZE_INSTRUCTION_MUTATION_TYPE, "
                "  RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE, "
                "  INSERT_INSTRUCTION_MUTATION_TYPE, "
                "  TRADE_INSTRUCTION_MUTATION_TYPE, "
                "  REMOVE_INSTRUCTION_MUTATION_TYPE "
                "] "),
            0.5,                                           // mutate_prob
            {NO_OP, SCALAR_SUM_OP, VECTOR_SUM_OP},         // allowed_setup_ops
            {NO_OP, SCALAR_DIFF_OP, VECTOR_DIFF_OP},       // allowed_predict_ops
            {NO_OP, SCALAR_PRODUCT_OP, VECTOR_PRODUCT_OP}, // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000,                  // min/max component function sizes
            &bit_gen,
            &rand_gen);
        Generator generator(NO_OP_ALGORITHM, 10, 10, 10, {}, {}, {}, nullptr,
                            nullptr);
        shared_ptr<const Algorithm> algorithm =
            make_shared<const Algorithm>(SimpleRandomAlgorithm());
        mutator.Mutate(&algorithm);
    }

    TEST(MutatorTest, CoversActions)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [ "
                "  INSERT_INSTRUCTION_MUTATION_TYPE, "
                "  TRADE_INSTRUCTION_MUTATION_TYPE, "
                "  REMOVE_INSTRUCTION_MUTATION_TYPE "
                "] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {NO_OP},                      // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen,
            &rand_gen);
        const Algorithm algorithm = SimpleRandomAlgorithm();
        const IntegerT original_size = algorithm.predict_.size();
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
                mutator.Mutate(&mutated_algorithm);
                return static_cast<IntegerT>(mutated_algorithm->predict_.size());
            }),
            {original_size - 1, original_size, original_size + 1},
            {original_size - 1, original_size, original_size + 1}));
    }

    TEST(MutatorTest, RespectsMutateProb)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [ "
                "  INSERT_INSTRUCTION_MUTATION_TYPE "
                "] "),
            0.5,                          // mutate_prob
            {},                           // allowed_setup_ops
            {NO_OP},                      // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen,
            &rand_gen);
        const Algorithm algorithm = SimpleRandomAlgorithm();
        const IntegerT original_size = algorithm.predict_.size();
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
                mutator.Mutate(&mutated_algorithm);
                return static_cast<IntegerT>(mutated_algorithm->predict_.size());
            }),
            {original_size, original_size + 1},
            {original_size, original_size + 1}));
    }

    TEST(MutatorTest, InstructionIndexTest)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0, {}, {}, {},
            0, 10000, 0, 10000, 0, 10000,
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<InstructionIndexT(void)>(
                [&]() { return mutator.RandomInstructionIndex(5); }),
            Range<InstructionIndexT>(0, 5),
            Range<InstructionIndexT>(0, 5)));
    }

    TEST(MutatorTest, SetupOpTest)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0,
            // allowed_setup_ops
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            {},                           // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<Op(void)>([&]() {
                return mutator.RandomSetupOp();
            }),
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}));
    }

    TEST(MutatorTest, PredictOpTest)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0,
            {}, // allowed_setup_ops
            // allowed_predict_ops
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<Op(void)>([&]() {
                return mutator.RandomPredictOp();
            }),
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}));
    }

    TEST(MutatorTest, LearnOpTest)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0,
            {}, // allowed_setup_ops
            {}, // allowed_predict_ops
            // allowed_learn_ops
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<Op(void)>([&]() {
                return mutator.RandomLearnOp();
            }),
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}));
    }

    TEST(MutatorTest, ComponentFunctionTest_SetupPredictLearn)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0,
            {NO_OP, SCALAR_SUM_OP},       // allowed_setup_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_predict_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<ComponentFunctionT(void)>([&]() {
                return mutator.RandomComponentFunction();
            }),
            {kSetupComponentFunction, kPredictComponentFunction,
             kLearnComponentFunction},
            {kSetupComponentFunction, kPredictComponentFunction,
             kLearnComponentFunction}));
    }

    TEST(MutatorTest, ComponentFunctionTest_Setup)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0,
            {NO_OP, SCALAR_SUM_OP},       // allowed_setup_ops
            {},                           // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<ComponentFunctionT(void)>([&]() {
                return mutator.RandomComponentFunction();
            }),
            {kSetupComponentFunction},
            {kSetupComponentFunction}));
    }

    TEST(MutatorTest, ComponentFunctionTest_Predict)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0,
            {},                           // allowed_setup_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<ComponentFunctionT(void)>([&]() {
                return mutator.RandomComponentFunction();
            }),
            {kPredictComponentFunction},
            {kPredictComponentFunction}));
    }

    TEST(MutatorTest, ComponentFunctionTest_Learn)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            MutationTypeList(), 0.0,
            {},                           // allowed_setup_ops
            {},                           // allowed_predict_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<ComponentFunctionT(void)>([&]() {
                return mutator.RandomComponentFunction();
            }),
            {kLearnComponentFunction},
            {kLearnComponentFunction}));
    }

    TEST(MutatorTest, AlterParam)
    {
        RandomGenerator rand_gen;
        const Algorithm algorithm = SimpleRandomAlgorithm();
        Mutator mutator;
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.AlterParam(&mutated_algorithm);
                return CountDifferentInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.AlterParam(&mutated_algorithm);
                return CountDifferentSetupInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.AlterParam(&mutated_algorithm);
                return CountDifferentPredictInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.AlterParam(&mutated_algorithm);
                return CountDifferentLearnInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
    }

    void addLoopInstructionHavingSingleInstructionInBody(ComponentFunction &componentFunction)
    {
        // for(s6 = 1..s5) {
        Instruction loop = Instruction(LOOP, 5, 6);
        //     s0 = s6 - s2
        loop.children_.emplace_back(std::make_shared<Instruction>(
            SCALAR_DIFF_OP,
            6,
            2,
            0));
        //     }
        componentFunction.getInstructions().emplace_back(std::make_shared<Instruction>(loop));
    }

    void addLoopInstructionEmptyBody(ComponentFunction &componentFunction)
    {
        // for(s6 = 1..s5) {
        Instruction loop = Instruction(LOOP, 5, 6);
        //     }
        componentFunction.getInstructions().emplace_back(std::make_shared<Instruction>(loop));
    }

    TEST(MutatorTest, AlterParamInLoopEmptyBody)
    {
        // Given
        Algorithm algorithm;
        addLoopInstructionEmptyBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [ ALTER_PARAM_MUTATION_TYPE ]"),
            1.0,
            {},                           // allowed_setup_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        Algorithm mutated_algorithm = algorithm;

        // When
        mutator.AlterParam(&mutated_algorithm);

        // Then
        EXPECT_NE(mutated_algorithm.predict_.getConstInstructions()[0], algorithm.predict_.getConstInstructions()[0]);
    }

    TEST(MutatorTest, AlterParamInLoop)
    {
        // Given
        Algorithm algorithm;
        addLoopInstructionHavingSingleInstructionInBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [ ALTER_PARAM_MUTATION_TYPE ]"),
            1.0,
            {},                           // allowed_setup_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<bool(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                // When
                mutator.AlterParam(&mutated_algorithm);
                auto mutatedInstructionOfLoopBody = mutated_algorithm.predict_.getConstInstructions()[0]->children_[0];
                auto instructionOfLoopBody = algorithm.predict_.getConstInstructions()[0]->children_[0];
                // Then
                return !mutatedInstructionOfLoopBody->isSameOpButHasDifferentParam(*instructionOfLoopBody);
            }),
            {true, false},
            {false}));
    }

    TEST(MutatorTest, RandomizeLoopInstruction)
    {
        // Given
        Algorithm algorithm;
        addLoopInstructionEmptyBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Op op = SCALAR_SUM_OP;
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [ RANDOMIZE_INSTRUCTION_MUTATION_TYPE ]"),
            1.0,
            {},                           // allowed_setup_ops
            {op},                         // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        Algorithm mutated_algorithm = algorithm;

        // When
        mutator.RandomizeInstruction(&mutated_algorithm);

        // Then
        EXPECT_EQ(mutated_algorithm.predict_.getConstInstructions()[0]->op_, op);
        EXPECT_TRUE(mutated_algorithm.predict_.getConstInstructions()[0]->children_.empty());
    }

    TEST(MutatorTest, RandomizeLoopInstruction_LoopBody)
    {
        // Given
        Algorithm algorithm;
        addLoopInstructionHavingSingleInstructionInBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [ RANDOMIZE_INSTRUCTION_MUTATION_TYPE ]"),
            1.0,
            {},                           // allowed_setup_ops
            {SCALAR_SUM_OP},              // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);

        auto loop = algorithm.predict_.getConstInstructions()[0];

        EXPECT_TRUE(IsEventually(
            function<bool(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                // When
                mutator.RandomizeInstruction(&mutated_algorithm);

                // Then
                auto mutatedLoop = mutated_algorithm.predict_.getConstInstructions()[0];
                bool sameOpAndSameParams = mutatedLoop->op_ == loop->op_ && mutatedLoop->paramEquals(*loop);

                return sameOpAndSameParams &&
                       mutatedLoop->children_.size() == loop->children_.size() &&
                       mutatedLoop->children_[0]->op_ == SCALAR_SUM_OP &&
                       *mutatedLoop->children_[0] != *loop->children_[0];
            }),
            {true, false},
            {true}));
    }

    TEST(MutatorTest, RandomizeInstruction)
    {
        RandomGenerator rand_gen;
        const Algorithm algorithm = SimpleRandomAlgorithm();
        Mutator mutator;
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeInstruction(&mutated_algorithm);
                return CountDifferentInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeInstruction(&mutated_algorithm);
                return CountDifferentSetupInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeInstruction(&mutated_algorithm);
                return CountDifferentPredictInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeInstruction(&mutated_algorithm);
                return CountDifferentLearnInstructions(mutated_algorithm, algorithm);
            }),
            {0, 1}, {1}));
    }

    TEST(MutatorTest, RandomizeComponentFunction)
    {
        RandomGenerator rand_gen;
        const Algorithm algorithm = SimpleRandomAlgorithm();
        Mutator mutator;
        const IntegerT setup_size = algorithm.setup_.size();
        const IntegerT predict_size = algorithm.predict_.size();
        const IntegerT learn_size = algorithm.learn_.size();
        vector<IntegerT> num_instr = {setup_size, predict_size, learn_size};
        const IntegerT max_instructions =
            *max_element(num_instr.begin(), num_instr.end());
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeComponentFunction(&mutated_algorithm);
                return CountDifferentInstructions(mutated_algorithm, algorithm);
            }),
            Range<IntegerT>(0, max_instructions + 1), {max_instructions}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeComponentFunction(&mutated_algorithm);
                return CountDifferentSetupInstructions(mutated_algorithm, algorithm);
            }),
            Range<IntegerT>(0, setup_size + 1),
            {setup_size}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeComponentFunction(&mutated_algorithm);
                return CountDifferentPredictInstructions(mutated_algorithm, algorithm);
            }),
            Range<IntegerT>(0, predict_size + 1),
            {predict_size}));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeComponentFunction(&mutated_algorithm);
                return CountDifferentLearnInstructions(mutated_algorithm, algorithm);
            }),
            Range<IntegerT>(0, learn_size + 1),
            {learn_size}));
    }

    TEST(MutatorTest, IdentityMutationType_WorksCorrectly)
    {
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        const Algorithm algorithm = SimpleRandomAlgorithm();
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [IDENTITY_MUTATION_TYPE] "),
            1.0,
            {NO_OP, SCALAR_SUM_OP},       // allowed_setup_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_predict_ops
            {NO_OP, SCALAR_SUM_OP},       // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
        mutator.Mutate(&mutated_algorithm);
        EXPECT_EQ(*mutated_algorithm, algorithm);
    }

    TEST(InsertInstructionMutationTypeTest, CoversComponentFunctions)
    {
        const Algorithm no_op_algorithm = SimpleRandomAlgorithm();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {SCALAR_SUM_OP},              // allowed_setup_ops
            {SCALAR_SUM_OP},              // allowed_predict_ops
            {SCALAR_SUM_OP},              // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, no_op_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
                mutator.Mutate(&mutated_algorithm);
                return DifferentComponentFunction(*mutated_algorithm, no_op_algorithm);
            }),
            {0, 1, 2}, {0, 1, 2}));
    }

    TEST(InsertInstructionMutationTypeTest, CoversSetupPositions)
    {
        const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
        const IntegerT component_function_size = no_op_algorithm.setup_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {SCALAR_SUM_OP},              // allowed_setup_ops
            {},                           // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        ASSERT_GT(component_function_size, 0);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, no_op_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
                mutator.Mutate(&mutated_algorithm);
                return ScalarSumOpPosition(mutated_algorithm->setup_.getConstInstructions());
            }),
            Range<IntegerT>(0, component_function_size + 1),
            Range<IntegerT>(0, component_function_size + 1),
            3));
    }

    TEST(InsertInstructionMutationTypeTest, CoversPredictPositions)
    {
        const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
        const IntegerT component_function_size = no_op_algorithm.predict_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {SCALAR_SUM_OP},              // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        ASSERT_GT(component_function_size, 0);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, no_op_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
                mutator.Mutate(&mutated_algorithm);
                return ScalarSumOpPosition(mutated_algorithm->predict_.getConstInstructions());
            }),
            Range<IntegerT>(0, component_function_size + 1),
            Range<IntegerT>(0, component_function_size + 1),
            3));
    }

    TEST(InsertInstructionMutationTypeTest, CoversLearnPositions)
    {
        const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
        const IntegerT component_function_size = no_op_algorithm.learn_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {},                           // allowed_predict_ops
            {SCALAR_SUM_OP},              // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        ASSERT_GT(component_function_size, 0);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, no_op_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
                mutator.Mutate(&mutated_algorithm);
                return ScalarSumOpPosition(mutated_algorithm->learn_.getConstInstructions());
            }),
            Range<IntegerT>(0, component_function_size + 1),
            Range<IntegerT>(0, component_function_size + 1),
            3));
    }

    TEST(InsertInstructionMutationTypeTest, InsertsWhenUnderMinSize)
    {
        const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
        const IntegerT setup_component_function_size = no_op_algorithm.setup_.size();
        const IntegerT predict_component_function_size = no_op_algorithm.predict_.size();
        const IntegerT learn_component_function_size = no_op_algorithm.learn_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                                // mutate_prob
            {SCALAR_SUM_OP},                    // allowed_setup_ops
            {SCALAR_SUM_OP},                    // allowed_predict_ops
            {SCALAR_SUM_OP},                    // allowed_learn_ops
            100, 10000, 100, 10000, 100, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
        mutator.Mutate(&mutated_algorithm);
        EXPECT_TRUE(mutated_algorithm->setup_.size() ==
                        setup_component_function_size + 1 ||
                    mutated_algorithm->predict_.size() ==
                        predict_component_function_size + 1 ||
                    mutated_algorithm->learn_.size() ==
                        learn_component_function_size + 1);
    }

    TEST(InsertInstructionMutationTypeTest, DoesNotInsertWhenOverMaxSize)
    {
        const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,              // mutate_prob
            {SCALAR_SUM_OP},  // allowed_setup_ops
            {SCALAR_SUM_OP},  // allowed_predict_ops
            {SCALAR_SUM_OP},  // allowed_learn_ops
            0, 1, 0, 1, 0, 1, // min/max component function sizes
            &bit_gen, &rand_gen);
        auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
        mutator.Mutate(&mutated_algorithm);
        EXPECT_EQ(*mutated_algorithm, no_op_algorithm);
    }

    TEST(InsertInstructionMutationTypeTest, CoversSetupInstructions)
    {
        const Algorithm empty_algorithm;
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                             // mutate_prob
            {SCALAR_SUM_OP, SCALAR_DIFF_OP}, // allowed_setup_ops
            {},                              // allowed_predict_ops
            {},                              // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000,    // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, empty_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(empty_algorithm);
                mutator.Mutate(&mutated_algorithm);
                if (mutated_algorithm->setup_.size() == 1)
                {
                    return static_cast<IntegerT>(mutated_algorithm->setup_.getConstInstructions()[0]->op_);
                }
                else
                {
                    return static_cast<IntegerT>(-1);
                }
            }),
            {SCALAR_SUM_OP, SCALAR_DIFF_OP}, {SCALAR_SUM_OP, SCALAR_DIFF_OP}, 3));
    }

    TEST(InsertInstructionMutationTypeTest, CoversPredictInstructions)
    {
        const Algorithm empty_algorithm;
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                             // mutate_prob
            {},                              // allowed_setup_ops
            {SCALAR_SUM_OP, SCALAR_DIFF_OP}, // allowed_predict_ops
            {},                              // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000,    // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, empty_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(empty_algorithm);
                mutator.Mutate(&mutated_algorithm);
                if (mutated_algorithm->predict_.size() == 1)
                {
                    return static_cast<IntegerT>(mutated_algorithm->predict_.getConstInstructions()[0]->op_);
                }
                else
                {
                    return static_cast<IntegerT>(-1);
                }
            }),
            {SCALAR_SUM_OP, SCALAR_DIFF_OP}, {SCALAR_SUM_OP, SCALAR_DIFF_OP}, 3));
    }

    TEST(InsertInstructionMutationTypeTest, InsertsInstructionInLoopBody)
    {
        Algorithm algorithm;
        addLoopInstructionEmptyBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {SCALAR_SUM_OP},              // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);

                // When
                mutator.Mutate(&mutated_algorithm);

                // Then
                if (mutated_algorithm->predict_.size() >= 1 && mutated_algorithm->predict_.getConstInstructions()[0]->op_ == LOOP)
                {
                    auto loop = mutated_algorithm->predict_.getConstInstructions()[0];
                    return loop->children_.size() >= 1 ? static_cast<IntegerT>(loop->children_[0]->op_) : static_cast<IntegerT>(-1);
                }
                else
                {
                    return static_cast<IntegerT>(-1);
                }
            }),
            {-1, SCALAR_SUM_OP},
            {SCALAR_SUM_OP},
            3));
    }

    TEST(InsertInstructionMutationTypeTest, InsertsInstructionBeforeLoop)
    {
        Algorithm algorithm;
        addLoopInstructionHavingSingleInstructionInBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {SCALAR_SUM_OP},              // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<bool(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);

                // When
                mutator.Mutate(&mutated_algorithm);

                // Then
                return mutated_algorithm->predict_.getConstInstructions().size() >= 2 &&
                       mutated_algorithm->predict_.getConstInstructions()[0]->op_ == SCALAR_SUM_OP &&
                       mutated_algorithm->predict_.getConstInstructions()[1]->op_ == LOOP;
            }),
            {false, true},
            {true},
            3));
    }

    TEST(InsertInstructionMutationTypeTest, InsertsInstructionAfterLoop)
    {
        Algorithm algorithm;
        addLoopInstructionHavingSingleInstructionInBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {SCALAR_SUM_OP},              // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<bool(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);

                // When
                mutator.Mutate(&mutated_algorithm);

                // Then
                return mutated_algorithm->predict_.getConstInstructions().size() >= 2 &&
                       mutated_algorithm->predict_.getConstInstructions()[0]->op_ == LOOP &&
                       mutated_algorithm->predict_.getConstInstructions()[1]->op_ == SCALAR_SUM_OP;
            }),
            {false, true},
            {true},
            3));
    }

    TEST(InsertInstructionMutationTypeTest, CoversLearnInstructions)
    {
        const Algorithm empty_algorithm;
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [INSERT_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                             // mutate_prob
            {},                              // allowed_setup_ops
            {},                              // allowed_predict_ops
            {SCALAR_SUM_OP, SCALAR_DIFF_OP}, // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000,    // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, empty_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(empty_algorithm);
                mutator.Mutate(&mutated_algorithm);
                if (mutated_algorithm->learn_.size() == 1)
                {
                    return static_cast<IntegerT>(mutated_algorithm->learn_.getConstInstructions()[0]->op_);
                }
                else
                {
                    return static_cast<IntegerT>(-1);
                }
            }),
            {SCALAR_SUM_OP, SCALAR_DIFF_OP}, {SCALAR_SUM_OP, SCALAR_DIFF_OP}, 3));
    }

    TEST(RemoveInstructionMutationTypeTest, RemovesLoopInstruction)
    {
        Algorithm algorithm;
        addLoopInstructionEmptyBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {NO_OP},                      // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);

                mutator.Mutate(&mutated_algorithm);

                bool loopInstructionRemoved = mutated_algorithm->predict_.empty();
                return loopInstructionRemoved;
            }),
            {false, true},
            {true},
            3));
    }

    TEST(RemoveInstructionMutationTypeTest, RemovesLoopBodyInstruction)
    {
        Algorithm algorithm;
        addLoopInstructionHavingSingleInstructionInBody(algorithm.predict_);
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>("mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {NO_OP},                      // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);

                mutator.Mutate(&mutated_algorithm);

                bool hasLoop = !mutated_algorithm->predict_.empty();
                if (!hasLoop)
                {
                    return false;
                }

                auto loop = mutated_algorithm->predict_.getConstInstructions()[0];
                bool isSingleInstructionInBodyRemoved = loop->children_.empty();
                return isSingleInstructionInBodyRemoved;
            }),
            {false, true},
            {true},
            3));
    }

    TEST(RemoveInstructionMutationTypeTest, CoversComponentFunctions)
    {
        const Algorithm random_algorithm = SimpleRandomAlgorithm();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0, {NO_OP}, {NO_OP}, {NO_OP},
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, random_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(random_algorithm);
                mutator.Mutate(&mutated_algorithm);
                return DifferentComponentFunction(*mutated_algorithm, random_algorithm);
            }),
            {0, 1, 2}, {0, 1, 2}, 3));
    }

    TEST(RemoveInstructionMutationTypeTest, CoversSetupPositions)
    {
        const Algorithm algorithm = SimpleIncreasingDataAlgorithm();
        const IntegerT component_function_size = algorithm.setup_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {NO_OP},                      // allowed_setup_ops
            {},                           // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        ASSERT_GT(component_function_size, 0);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
                mutator.Mutate(&mutated_algorithm);
                return MissingDataInComponentFunction(
                    algorithm.setup_.getConstInstructions(), mutated_algorithm->setup_.getConstInstructions());
            }),
            Range<IntegerT>(0, component_function_size),
            Range<IntegerT>(0, component_function_size),
            3));
    }

    TEST(RemoveInstructionMutationTypeTest, CoversPredictPositions)
    {
        const Algorithm algorithm = SimpleIncreasingDataAlgorithm();
        const IntegerT component_function_size = algorithm.predict_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {NO_OP},                      // allowed_predict_ops
            {},                           // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        ASSERT_GT(component_function_size, 0);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);

                mutator.Mutate(&mutated_algorithm);
                return MissingDataInComponentFunction(
                    algorithm.predict_.getConstInstructions(), mutated_algorithm->predict_.getConstInstructions());
            }),
            Range<IntegerT>(0, component_function_size),
            Range<IntegerT>(0, component_function_size),
            3));
    }

    TEST(RemoveInstructionMutationTypeTest, CoversLearnPositions)
    {
        const Algorithm algorithm = SimpleIncreasingDataAlgorithm();
        const IntegerT component_function_size = algorithm.learn_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                          // mutate_prob
            {},                           // allowed_setup_ops
            {},                           // allowed_predict_ops
            {NO_OP},                      // allowed_learn_ops
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        ASSERT_GT(component_function_size, 0);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
                mutator.Mutate(&mutated_algorithm);
                return MissingDataInComponentFunction(
                    algorithm.learn_.getConstInstructions(), mutated_algorithm->learn_.getConstInstructions());
            }),
            Range<IntegerT>(0, component_function_size),
            Range<IntegerT>(0, component_function_size),
            3));
    }

    TEST(RemoveInstructionMutationTypeTest, DoesNotRemoveWhenUnderMinSize)
    {
        const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0,                                // mutate_prob
            {SCALAR_SUM_OP},                    // allowed_setup_ops
            {SCALAR_SUM_OP},                    // allowed_predict_ops
            {SCALAR_SUM_OP},                    // allowed_learn_ops
            100, 10000, 100, 10000, 100, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
        mutator.Mutate(&mutated_algorithm);
        EXPECT_EQ(*mutated_algorithm, no_op_algorithm);
    }

    TEST(RemoveInstructionMutationTypeTest, RemovesWhenOverMaxSize)
    {
        const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
        const IntegerT setup_component_function_size = no_op_algorithm.setup_.size();
        const IntegerT predict_component_function_size =
            no_op_algorithm.predict_.size();
        const IntegerT learn_component_function_size = no_op_algorithm.learn_.size();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [REMOVE_INSTRUCTION_MUTATION_TYPE] "),
            1.0,              // mutate_prob
            {SCALAR_SUM_OP},  // allowed_setup_ops
            {SCALAR_SUM_OP},  // allowed_predict_ops
            {SCALAR_SUM_OP},  // allowed_learn_ops
            0, 1, 0, 1, 0, 1, // min/max component function sizes
            &bit_gen, &rand_gen);
        auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
        mutator.Mutate(&mutated_algorithm);
        EXPECT_TRUE(mutated_algorithm->setup_.size() ==
                        setup_component_function_size - 1 ||
                    mutated_algorithm->predict_.size() ==
                        predict_component_function_size - 1 ||
                    mutated_algorithm->learn_.size() ==
                        learn_component_function_size - 1);
    }

    TEST(TradeInstructionMutationTypeTest, CoversComponentFunctions)
    {
        const Algorithm random_algorithm = SimpleRandomAlgorithm();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [TRADE_INSTRUCTION_MUTATION_TYPE] "),
            1.0, {NO_OP}, {NO_OP}, {NO_OP},
            0, 10000, 0, 10000, 0, 10000, // min/max component function sizes
            &bit_gen, &rand_gen);
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&mutator, random_algorithm]() {
                auto mutated_algorithm = make_shared<const Algorithm>(random_algorithm);
                mutator.Mutate(&mutated_algorithm);
                return DifferentComponentFunction(*mutated_algorithm, random_algorithm);
            }),
            {-1, 0, 1, 2}, {0, 1, 2}, 3));
    }

    TEST(TradeInstructionMutationTypeTest, PreservesSizes)
    {
        const Algorithm random_algorithm = SimpleRandomAlgorithm();
        mt19937 bit_gen;
        RandomGenerator rand_gen(&bit_gen);
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [TRADE_INSTRUCTION_MUTATION_TYPE] "),
            1.0, {NO_OP}, {NO_OP}, {NO_OP},
            0, 10000, 0, 10000, 0, 10000, // min/max prog. sizes
            &bit_gen, &rand_gen);
        auto mutated_algorithm = make_shared<const Algorithm>(random_algorithm);
        mutator.Mutate(&mutated_algorithm);
        EXPECT_EQ(mutated_algorithm->setup_.size(), random_algorithm.setup_.size());
        EXPECT_EQ(mutated_algorithm->predict_.size(), random_algorithm.predict_.size());
        EXPECT_EQ(mutated_algorithm->learn_.size(), random_algorithm.learn_.size());
    }

    TEST(MutatorTest, RandomizeAlgorithm)
    {
        RandomGenerator rand_gen;
        mt19937 bit_gen;
        const Algorithm algorithm = SimpleRandomAlgorithm();
        const IntegerT setup_size = algorithm.setup_.size();
        const IntegerT predict_size = algorithm.predict_.size();
        const IntegerT learn_size = algorithm.learn_.size();
        Mutator mutator(
            ParseTextFormat<MutationTypeList>(
                "mutation_types: [RANDOMIZE_ALGORITHM_MUTATION_TYPE] "),
            1.0,
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
            setup_size, setup_size + 1,
            predict_size, predict_size + 1,
            learn_size, learn_size + 1,
            &bit_gen, &rand_gen);
        const IntegerT num_instr = setup_size + predict_size + learn_size;
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&]() {
                Algorithm mutated_algorithm = algorithm;
                mutator.RandomizeAlgorithm(&mutated_algorithm);
                return CountDifferentInstructions(mutated_algorithm, algorithm);
            }),
            Range<IntegerT>(0, num_instr + 1), {num_instr}));
    }

} // namespace automl_zero
