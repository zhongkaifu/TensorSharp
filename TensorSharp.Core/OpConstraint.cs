// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;

namespace TensorSharp
{
    public abstract class OpConstraint
    {
        public abstract bool SatisfiedFor(object?[] args);
    }

    public class ArgCountConstraint : OpConstraint
    {
        private readonly int argCount;

        public ArgCountConstraint(int argCount) { this.argCount = argCount; }

        public override bool SatisfiedFor(object?[] args)
        {
            return args.Length == argCount;
        }
    }

    public class ArgStorageTypeConstraint : OpConstraint
    {
        private readonly int argIndex;
        private readonly Type requiredType;
        private readonly bool allowNull;

        public ArgStorageTypeConstraint(int argIndex, Type requiredType, bool allowNull = true)
        {
            this.argIndex = argIndex;
            this.requiredType = requiredType;
            this.allowNull = allowNull;
        }

        public override bool SatisfiedFor(object?[] args)
        {
            if (allowNull && args[argIndex] == null)
            {
                return true;
            }
            else if (!allowNull && args[argIndex] == null)
            {
                return false;
            }

            // it could be that args[argIndex] is null and allowNull is true,
            Storage? argStorage = ((Tensor?)args[argIndex])?.Storage;
            return argStorage?.GetType() == requiredType;
        }
    }
}
