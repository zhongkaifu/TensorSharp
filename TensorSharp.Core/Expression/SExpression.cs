// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System;

namespace TensorSharp.Expression
{
    public abstract class SExpression
    {
        public abstract float Evaluate();
    }


    public class ConstScalarExpression : SExpression
    {
        private readonly float value;

        public ConstScalarExpression(float value)
        {
            this.value = value;
        }

        public override float Evaluate()
        {
            return value;
        }
    }

    public class DelegateScalarExpression : SExpression
    {
        private readonly Func<float> evaluate;

        public DelegateScalarExpression(Func<float> evaluate)
        {
            this.evaluate = evaluate;
        }

        public override float Evaluate()
        {
            return evaluate();
        }
    }

    public class UnaryScalarExpression : SExpression
    {
        private readonly SExpression src;
        private readonly Func<float, float> evaluate;


        public UnaryScalarExpression(SExpression src, Func<float, float> evaluate)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        public override float Evaluate()
        {
            return evaluate(src.Evaluate());
        }
    }

    public class BinaryScalarExpression : SExpression
    {
        private readonly SExpression left;
        private readonly SExpression right;
        private readonly Func<float, float, float> evaluate;


        public BinaryScalarExpression(SExpression left, SExpression right, Func<float, float, float> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        public override float Evaluate()
        {
            return evaluate(left.Evaluate(), right.Evaluate());
        }
    }
}
