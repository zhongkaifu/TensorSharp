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
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace TensorSharp
{
    [AttributeUsage(AttributeTargets.Class)]
    public class OpsClassAttribute : Attribute
    {
    }

    [AttributeUsage(AttributeTargets.Method)]
    public abstract class RegisterOp : Attribute
    {
        public string OpName { get; private set; }

        public RegisterOp(string opName)
        {
            OpName = opName;
        }

        protected static object InvokeRegisteredMethod(object instance, MethodInfo method, object[] args)
        {
            try
            {
                return method.Invoke(instance, args);
            }
            catch (TargetInvocationException ex) when (ex.InnerException != null)
            {
                ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }

        public abstract void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints);
    }

    /// <summary>
    /// Register a method where the only constraint is that the argument counts match.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpArgCount : RegisterOp
    {
        public RegisterOpArgCount(string opName) : base(opName)
        {
        }

        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            List<OpConstraint> constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            OpRegistry.Register(OpName, args => InvokeRegisteredMethod(instance, method, args), constraints);
        }
    }


    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpStorageType : RegisterOp
    {
        private readonly Type storageType;

        public RegisterOpStorageType(string opName, Type storageType) : base(opName)
        {
            this.storageType = storageType;
        }

        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            List<OpConstraint> constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            ParameterInfo[] methodParams = method.GetParameters();
            for (int i = 0; i < methodParams.Length; ++i)
            {
                if (methodParams[i].ParameterType == typeof(Tensor))
                {
                    constraints.Add(new ArgStorageTypeConstraint(i, storageType));
                }
            }

            OpRegistry.Register(OpName, args => InvokeRegisteredMethod(instance, method, args), constraints);
        }
    }




    [AttributeUsage(AttributeTargets.Parameter)]
    public abstract class ArgConstraintAttribute : Attribute
    {
        public ArgConstraintAttribute()
        {
        }

        public abstract IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance);
    }

    [AttributeUsage(AttributeTargets.Parameter)]
    public class OpArgStorageType : ArgConstraintAttribute
    {
        private readonly Type storageType;

        public OpArgStorageType(Type storageType)
        {
            this.storageType = storageType;
        }

        public override IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance)
        {
            yield return new ArgStorageTypeConstraint(parameter.Position, storageType);
        }
    }
}
