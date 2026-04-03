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
using System.Linq;
using System.Reflection;

namespace TensorSharp
{
    public delegate object OpHandler(object[] args);

    public static class OpRegistry
    {
        private class OpInstance
        {
            public OpHandler handler;
            public IEnumerable<OpConstraint> constraints;
        }

        private static readonly Dictionary<string, List<OpInstance>> opInstances = new Dictionary<string, List<OpInstance>>();
        // Remember which assemblies have been registered to avoid accidental double-registering
        private static readonly HashSet<Assembly> registeredAssemblies = new HashSet<Assembly>();

        static OpRegistry()
        {
            // Register CPU ops from this assembly
            RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public static void Register(string opName, OpHandler handler, IEnumerable<OpConstraint> constraints)
        {
            OpInstance newInstance = new OpInstance() { handler = handler, constraints = constraints };

            if (opInstances.TryGetValue(opName, out List<OpInstance> instanceList))
            {
                instanceList.Add(newInstance);
            }
            else
            {
                instanceList = new List<OpInstance>
                {
                    newInstance
                };
                opInstances.Add(opName, instanceList);
            }
        }

        public static object Invoke(string opName, params object[] args)
        {
            if (opInstances.TryGetValue(opName, out List<OpInstance> instanceList))
            {
                foreach (OpInstance instance in instanceList)
                {
                    if (instance.constraints.All(x => x.SatisfiedFor(args)))
                    {
                        return instance.handler.Invoke(args);
                    }
                }

                throw new ApplicationException("None of the registered handlers match the arguments for " + opName);
            }
            else
            {
                throw new ApplicationException("No handlers have been registered for op " + opName);
            }
        }

        public static void RegisterAssembly(Assembly assembly)
        {
            if (!registeredAssemblies.Contains(assembly))
            {
                registeredAssemblies.Add(assembly);

                IEnumerable<Type> types = assembly.TypesWithAttribute<OpsClassAttribute>(false)
                    .Select(x => x.Item1);

                foreach (Type type in types)
                {
                    object instance = Activator.CreateInstance(type);

                    IEnumerable<Tuple<MethodInfo, IEnumerable<RegisterOp>>> methods = type.MethodsWithAttribute<RegisterOp>(false);
                    foreach (Tuple<MethodInfo, IEnumerable<RegisterOp>> method in methods)
                    {
                        IEnumerable<OpConstraint> paramConstraints = GetParameterConstraints(method.Item1, instance);
                        foreach (RegisterOp attribute in method.Item2)
                        {
                            attribute.DoRegister(instance, method.Item1, paramConstraints);
                        }
                    }
                }
            }
        }

        private static IEnumerable<OpConstraint> GetParameterConstraints(MethodInfo method, object instance)
        {
            IEnumerable<OpConstraint> result = Enumerable.Empty<OpConstraint>();
            foreach (Tuple<ParameterInfo, IEnumerable<ArgConstraintAttribute>> parameter in method.ParametersWithAttribute<ArgConstraintAttribute>(false))
            {
                foreach (ArgConstraintAttribute attribute in parameter.Item2)
                {
                    result = Enumerable.Concat(result, attribute.GetConstraints(parameter.Item1, instance));
                }
            }

            return result;
        }
    }
}
