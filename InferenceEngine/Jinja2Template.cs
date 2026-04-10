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
using System.Globalization;
using System.Linq;
using System.Text;

namespace InferenceEngine
{
    /// <summary>
    /// Minimal Jinja2 template renderer for LLM chat templates loaded from GGUF files.
    /// Supports: for/endfor, if/elif/else/endif, set, {{ output }}, {# comments #},
    /// whitespace control ({%- / -%}), dict/array access, comparisons, boolean logic,
    /// filters (trim, length, default, first, last, upper, lower), method calls (.items(), .get()),
    /// string concatenation, loop variables (loop.first, loop.last, loop.index, loop.index0),
    /// array slicing, 'in' operator, 'is defined' / 'is not defined', ternary expressions.
    /// </summary>
    public sealed class Jinja2Template
    {
        #region Segment / AST types

        private enum SegKind { Text, Output, Block }

        private readonly struct Seg
        {
            public readonly SegKind Kind;
            public readonly string Body;
            public readonly bool TrimBefore;
            public readonly bool TrimAfter;

            public Seg(SegKind kind, string body, bool trimBefore, bool trimAfter)
            {
                Kind = kind;
                Body = body;
                TrimBefore = trimBefore;
                TrimAfter = trimAfter;
            }
        }

        private abstract class Node { }
        private sealed class TextNode : Node { public string Text; }
        private sealed class OutputNode : Node { public string Expr; }
        private sealed class SetNode : Node { public string VarName; public string ValueExpr; }
        private sealed class ForNode : Node
        {
            public string VarName;
            public string VarName2; // optional second var for tuple unpacking (key, value)
            public string IterExpr;
            public List<Node> Body;
        }
        private sealed class IfNode : Node
        {
            public List<(string Cond, List<Node> Body)> Branches;
        }
        private sealed class MacroNode : Node
        {
            public string Name;
            public string Args;
            public List<Node> Body;
        }

        #endregion

        private readonly List<Node> _nodes;

        public Jinja2Template(string template)
        {
            _nodes = Parse(Tokenize(template));
        }

        public string Render(Dictionary<string, object> context)
        {
            var sb = new StringBuilder();
            RenderNodes(_nodes, new Context(context), sb);
            return sb.ToString();
        }

        #region Tokenizer

        private static List<Seg> Tokenize(string tmpl)
        {
            var segs = new List<Seg>();
            int i = 0;

            while (i < tmpl.Length)
            {
                int next = FindTag(tmpl, i);
                if (next < 0)
                {
                    segs.Add(new Seg(SegKind.Text, tmpl.Substring(i), false, false));
                    break;
                }

                if (next > i)
                    segs.Add(new Seg(SegKind.Text, tmpl.Substring(i, next - i), false, false));

                char c2 = tmpl[next + 1];
                if (c2 == '#')
                {
                    int end = tmpl.IndexOf("#}", next + 2, StringComparison.Ordinal);
                    if (end < 0) throw new FormatException("Unclosed comment tag.");
                    bool tb = next + 2 < tmpl.Length && tmpl[next + 2] == '-';
                    bool ta = end >= 2 && tmpl[end - 1] == '-';
                    segs.Add(new Seg(SegKind.Text, "", tb, ta));
                    i = end + 2;
                }
                else
                {
                    string closeTag = c2 == '{' ? "}}" : "%}";
                    SegKind kind = c2 == '{' ? SegKind.Output : SegKind.Block;
                    int bodyStart = next + 2;
                    bool tb = bodyStart < tmpl.Length && tmpl[bodyStart] == '-';
                    if (tb) bodyStart++;

                    int end = tmpl.IndexOf(closeTag, bodyStart, StringComparison.Ordinal);
                    if (end < 0) throw new FormatException($"Unclosed {closeTag} tag.");
                    bool ta = end > 0 && tmpl[end - 1] == '-';
                    int bodyEnd = ta ? end - 1 : end;

                    string body = tmpl.Substring(bodyStart, bodyEnd - bodyStart).Trim();
                    segs.Add(new Seg(kind, body, tb, ta));
                    i = end + 2;
                }
            }

            ApplyTrimming(segs);
            return segs;
        }

        private static int FindTag(string s, int from)
        {
            for (int i = from; i < s.Length - 1; i++)
            {
                if (s[i] == '{' && (s[i + 1] == '{' || s[i + 1] == '%' || s[i + 1] == '#'))
                    return i;
            }
            return -1;
        }

        private static void ApplyTrimming(List<Seg> segs)
        {
            for (int i = 0; i < segs.Count; i++)
            {
                if (segs[i].Kind != SegKind.Text) continue;
                string t = segs[i].Body;

                if (i + 1 < segs.Count && segs[i + 1].TrimBefore)
                    t = t.TrimEnd();
                if (i > 0 && segs[i - 1].TrimAfter)
                    t = t.TrimStart();

                segs[i] = new Seg(SegKind.Text, t, segs[i].TrimBefore, segs[i].TrimAfter);
            }
        }

        #endregion

        #region Parser

        private static List<Node> Parse(List<Seg> segs)
        {
            int pos = 0;
            return ParseNodes(segs, ref pos, null);
        }

        private static List<Node> ParseNodes(List<Seg> segs, ref int pos, string endTag)
        {
            var nodes = new List<Node>();

            while (pos < segs.Count)
            {
                var seg = segs[pos];

                if (seg.Kind == SegKind.Text)
                {
                    if (seg.Body.Length > 0)
                        nodes.Add(new TextNode { Text = seg.Body });
                    pos++;
                    continue;
                }

                if (seg.Kind == SegKind.Output)
                {
                    nodes.Add(new OutputNode { Expr = seg.Body });
                    pos++;
                    continue;
                }

                string body = seg.Body;

                if (endTag != null && MatchesEndTag(body, endTag))
                    return nodes;

                if (body.StartsWith("macro "))
                {
                    pos++;
                    var macroNode = ParseMacro(body, segs, ref pos);
                    nodes.Add(macroNode);
                }
                else if (body.StartsWith("for "))
                {
                    pos++;
                    var forNode = ParseFor(body, segs, ref pos);
                    nodes.Add(forNode);
                }
                else if (body.StartsWith("if ") || body == "if")
                {
                    pos++;
                    var ifNode = ParseIf(body, segs, ref pos);
                    nodes.Add(ifNode);
                }
                else if (body.StartsWith("set "))
                {
                    var setNode = ParseSet(body);
                    nodes.Add(setNode);
                    pos++;
                }
                else if (body == "endif" || body == "endfor" || body == "endmacro" ||
                         body.StartsWith("elif ") || body == "elif" || body == "else")
                {
                    return nodes;
                }
                else
                {
                    pos++;
                }
            }

            return nodes;
        }

        private static bool MatchesEndTag(string body, string tag)
        {
            return body == tag || (tag == "endif" && (body == "elif" || body.StartsWith("elif ") || body == "else"))
                || (tag == "endfor" && body == "endfor")
                || (tag == "endmacro" && body == "endmacro");
        }

        private static ForNode ParseFor(string header, List<Seg> segs, ref int pos)
        {
            // "for x in expr" or "for k, v in expr"
            string rest = header.Substring(4).Trim();
            int inIdx = FindKeyword(rest, " in ");
            if (inIdx < 0) throw new FormatException($"Invalid for: {header}");

            string varPart = rest.Substring(0, inIdx).Trim();
            string iterExpr = rest.Substring(inIdx + 4).Trim();

            string var1 = varPart, var2 = null;
            int comma = varPart.IndexOf(',');
            if (comma >= 0)
            {
                var1 = varPart.Substring(0, comma).Trim();
                var2 = varPart.Substring(comma + 1).Trim();
            }

            var body = ParseNodes(segs, ref pos, "endfor");
            if (pos < segs.Count && segs[pos].Body == "endfor") pos++;

            return new ForNode { VarName = var1, VarName2 = var2, IterExpr = iterExpr, Body = body };
        }

        private static MacroNode ParseMacro(string header, List<Seg> segs, ref int pos)
        {
            // "macro name(args)"
            string rest = header.Substring(6).Trim();
            string name = rest, args = "";
            int paren = rest.IndexOf('(');
            if (paren >= 0)
            {
                name = rest.Substring(0, paren).Trim();
                int close = rest.IndexOf(')', paren);
                args = close > paren ? rest.Substring(paren + 1, close - paren - 1) : "";
            }

            var body = ParseNodes(segs, ref pos, "endmacro");
            if (pos < segs.Count && segs[pos].Body == "endmacro") pos++;

            return new MacroNode { Name = name, Args = args, Body = body };
        }

        private static int FindKeyword(string s, string kw)
        {
            int idx = 0;
            while (true)
            {
                int found = s.IndexOf(kw, idx, StringComparison.Ordinal);
                if (found < 0) return -1;
                // make sure it's not inside quotes
                if (!InsideQuotes(s, found)) return found;
                idx = found + 1;
            }
        }

        private static bool InsideQuotes(string s, int pos)
        {
            bool inSingle = false, inDouble = false;
            for (int i = 0; i < pos; i++)
            {
                if (s[i] == '\'' && !inDouble) inSingle = !inSingle;
                else if (s[i] == '"' && !inSingle) inDouble = !inDouble;
            }
            return inSingle || inDouble;
        }

        private static IfNode ParseIf(string header, List<Seg> segs, ref int pos)
        {
            string cond = header.Length > 3 ? header.Substring(3).Trim() : "true";
            var branches = new List<(string, List<Node>)>();

            var body = ParseNodes(segs, ref pos, "endif");
            branches.Add((cond, body));

            while (pos < segs.Count)
            {
                string tag = segs[pos].Body;
                if (tag == "endif") { pos++; break; }
                if (tag == "else")
                {
                    pos++;
                    body = ParseNodes(segs, ref pos, "endif");
                    branches.Add((null, body));
                    if (pos < segs.Count && segs[pos].Body == "endif") pos++;
                    break;
                }
                if (tag.StartsWith("elif ") || tag == "elif")
                {
                    string elifCond = tag.Length > 5 ? tag.Substring(5).Trim() : "true";
                    pos++;
                    body = ParseNodes(segs, ref pos, "endif");
                    branches.Add((elifCond, body));
                }
                else
                {
                    break;
                }
            }

            return new IfNode { Branches = branches };
        }

        private static SetNode ParseSet(string body)
        {
            // "set var = expr"
            string rest = body.Substring(4).Trim();
            int eq = rest.IndexOf('=');
            if (eq < 0) throw new FormatException($"Invalid set: {body}");
            return new SetNode
            {
                VarName = rest.Substring(0, eq).Trim(),
                ValueExpr = rest.Substring(eq + 1).Trim()
            };
        }

        #endregion

        #region Renderer

        private sealed class Context
        {
            private readonly List<Dictionary<string, object>> _scopes = new();

            public Context(Dictionary<string, object> global)
            {
                _scopes.Add(global);
            }

            public void Push() => _scopes.Add(new Dictionary<string, object>());
            public void Pop() => _scopes.RemoveAt(_scopes.Count - 1);

            public void Set(string name, object value) =>
                _scopes[_scopes.Count - 1][name] = value;

            public bool TryGet(string name, out object value)
            {
                for (int i = _scopes.Count - 1; i >= 0; i--)
                {
                    if (_scopes[i].TryGetValue(name, out value))
                        return true;
                }
                value = null;
                return false;
            }

            public object Get(string name) =>
                TryGet(name, out var v) ? v : null;

            public bool IsDefined(string name)
            {
                for (int i = _scopes.Count - 1; i >= 0; i--)
                    if (_scopes[i].ContainsKey(name)) return true;
                return false;
            }
        }

        private static void RenderNodes(List<Node> nodes, Context ctx, StringBuilder sb)
        {
            foreach (var node in nodes)
            {
                switch (node)
                {
                    case TextNode tn:
                        sb.Append(tn.Text);
                        break;
                    case OutputNode on:
                        var val = EvalExpr(on.Expr, ctx);
                        sb.Append(Stringify(val));
                        break;
                    case SetNode sn:
                    {
                        object setVal = EvalExpr(sn.ValueExpr, ctx);
                        int dotIdx = sn.VarName.IndexOf('.');
                        if (dotIdx > 0)
                        {
                            string nsName = sn.VarName.Substring(0, dotIdx);
                            string attrName = sn.VarName.Substring(dotIdx + 1);
                            if (ctx.Get(nsName) is IDictionary<string, object> nsDict)
                                nsDict[attrName] = setVal;
                        }
                        else
                        {
                            ctx.Set(sn.VarName, setVal);
                        }
                        break;
                    }
                    case ForNode fn:
                        RenderFor(fn, ctx, sb);
                        break;
                    case IfNode ifn:
                        RenderIf(ifn, ctx, sb);
                        break;
                    case MacroNode mn:
                        ctx.Set(mn.Name, mn);
                        break;
                }
            }
        }

        private static void RenderFor(ForNode fn, Context ctx, StringBuilder sb)
        {
            var iterable = EvalExpr(fn.IterExpr, ctx);
            var items = ToList(iterable);
            int count = items.Count;

            ctx.Push();
            for (int i = 0; i < count; i++)
            {
                var item = items[i];

                if (fn.VarName2 != null && item is KeyValuePair<string, object> kvp)
                {
                    ctx.Set(fn.VarName, kvp.Key);
                    ctx.Set(fn.VarName2, kvp.Value);
                }
                else if (fn.VarName2 != null && item is IList<object> tuple && tuple.Count >= 2)
                {
                    ctx.Set(fn.VarName, tuple[0]);
                    ctx.Set(fn.VarName2, tuple[1]);
                }
                else
                {
                    ctx.Set(fn.VarName, item);
                }

                var loopVar = new Dictionary<string, object>
                {
                    ["index0"] = i,
                    ["index"] = i + 1,
                    ["first"] = i == 0,
                    ["last"] = i == count - 1,
                    ["length"] = count,
                    ["revindex0"] = count - 1 - i,
                    ["revindex"] = count - i,
                };
                if (i > 0) loopVar["previtem"] = items[i - 1];
                if (i < count - 1) loopVar["nextitem"] = items[i + 1];
                ctx.Set("loop", loopVar);

                RenderNodes(fn.Body, ctx, sb);
            }
            ctx.Pop();
        }

        private static void RenderIf(IfNode ifn, Context ctx, StringBuilder sb)
        {
            foreach (var (cond, body) in ifn.Branches)
            {
                if (cond == null || Truthy(EvalExpr(cond, ctx)))
                {
                    RenderNodes(body, ctx, sb);
                    return;
                }
            }
        }

        /// <summary>
        /// Invoke a Jinja2 macro: parse parameter declarations, bind arguments, render body.
        /// </summary>
        private static string CallMacro(MacroNode macro, string callArgsStr, Context ctx)
        {
            // Parse macro parameter declarations: "param1, param2=default_value, ..."
            var paramDefs = SplitArgs(macro.Args);
            var paramNames = new List<string>();
            var paramDefaults = new Dictionary<string, string>();
            foreach (var pd in paramDefs)
            {
                int eq = pd.IndexOf('=');
                if (eq > 0)
                {
                    string pname = pd.Substring(0, eq).Trim();
                    string pdefault = pd.Substring(eq + 1).Trim();
                    paramNames.Add(pname);
                    paramDefaults[pname] = pdefault;
                }
                else
                {
                    paramNames.Add(pd.Trim());
                }
            }

            // Parse call arguments: positional and named
            var callArgs = callArgsStr.Length > 0 ? SplitArgs(callArgsStr) : new List<string>();
            var namedArgs = new Dictionary<string, string>();
            var positionalArgs = new List<string>();
            foreach (var ca in callArgs)
            {
                int eq = ca.IndexOf('=');
                if (eq > 0 && !ca.Substring(0, eq).Trim().StartsWith("'") &&
                    !ca.Substring(0, eq).Trim().StartsWith("\"") &&
                    ca.Substring(0, eq).Trim().All(c => char.IsLetterOrDigit(c) || c == '_'))
                {
                    namedArgs[ca.Substring(0, eq).Trim()] = ca.Substring(eq + 1).Trim();
                }
                else
                {
                    positionalArgs.Add(ca);
                }
            }

            // Bind parameters to arguments in a new scope
            ctx.Push();
            for (int j = 0; j < paramNames.Count; j++)
            {
                string pname = paramNames[j];
                if (namedArgs.TryGetValue(pname, out string namedExpr))
                {
                    ctx.Set(pname, EvalExpr(namedExpr, ctx));
                }
                else if (j < positionalArgs.Count)
                {
                    ctx.Set(pname, EvalExpr(positionalArgs[j], ctx));
                }
                else if (paramDefaults.TryGetValue(pname, out string defaultExpr))
                {
                    ctx.Set(pname, EvalExpr(defaultExpr, ctx));
                }
                else
                {
                    ctx.Set(pname, null);
                }
            }

            var sb = new StringBuilder();
            RenderNodes(macro.Body, ctx, sb);
            ctx.Pop();
            return sb.ToString();
        }

        #endregion

        #region Expression Evaluator

        private static object EvalExpr(string expr, Context ctx)
        {
            expr = expr.Trim();
            if (expr.Length == 0) return null;

            // Ternary: value_expr if cond_expr [else else_expr]
            int ifIdx = FindTopLevelKeyword(expr, " if ");
            if (ifIdx > 0)
            {
                string valueExpr = expr.Substring(0, ifIdx);
                string rest = expr.Substring(ifIdx + 4);
                int elseIdx = FindTopLevelKeyword(rest, " else ");
                if (elseIdx >= 0)
                {
                    string condExpr = rest.Substring(0, elseIdx);
                    string elseExpr = rest.Substring(elseIdx + 6);
                    return Truthy(EvalExpr(condExpr, ctx))
                        ? EvalExpr(valueExpr, ctx)
                        : EvalExpr(elseExpr, ctx);
                }
                else
                {
                    return Truthy(EvalExpr(rest, ctx))
                        ? EvalExpr(valueExpr, ctx)
                        : "";
                }
            }

            return EvalOr(expr, ctx);
        }

        private static object EvalOr(string expr, Context ctx)
        {
            var parts = SplitTopLevel(expr, " or ");
            if (parts.Count == 1) return EvalAnd(parts[0], ctx);
            foreach (string p in parts)
                if (Truthy(EvalAnd(p, ctx))) return true;
            return false;
        }

        private static object EvalAnd(string expr, Context ctx)
        {
            var parts = SplitTopLevel(expr, " and ");
            if (parts.Count == 1) return EvalNot(parts[0], ctx);
            foreach (string p in parts)
                if (!Truthy(EvalNot(p, ctx))) return false;
            return true;
        }

        private static object EvalNot(string expr, Context ctx)
        {
            expr = expr.Trim();
            if (expr.StartsWith("not "))
                return !Truthy(EvalNot(expr.Substring(4), ctx));
            return EvalComparison(expr, ctx);
        }

        private static object EvalComparison(string expr, Context ctx)
        {
            // Handle 'is' type tests: defined, mapping, string, boolean, sequence, none, etc.
            int isIdx = FindTopLevelKeyword(expr, " is ");
            if (isIdx > 0)
            {
                string lhs = expr.Substring(0, isIdx).Trim();
                string rhs = expr.Substring(isIdx + 4).Trim();
                bool negated = rhs.StartsWith("not ");
                if (negated) rhs = rhs.Substring(4).Trim();
                bool result = EvalIsTest(lhs, rhs, ctx);
                return negated ? !result : result;
            }

            // Handle 'in' operator
            int inIdx = FindTopLevelKeyword(expr, " in ");
            if (inIdx > 0)
            {
                var needle = EvalPrimary(expr.Substring(0, inIdx).Trim(), ctx);
                var haystack = EvalPrimary(expr.Substring(inIdx + 4).Trim(), ctx);
                return ContainsValue(haystack, needle);
            }

            // Handle 'not in' operator
            int notInIdx = FindTopLevelKeyword(expr, " not in ");
            if (notInIdx > 0)
            {
                var needle = EvalPrimary(expr.Substring(0, notInIdx).Trim(), ctx);
                var haystack = EvalPrimary(expr.Substring(notInIdx + 8).Trim(), ctx);
                return !ContainsValue(haystack, needle);
            }

            string[] cmpOps = { "==", "!=", ">=", "<=", ">", "<" };
            foreach (string op in cmpOps)
            {
                int idx = FindTopLevelOp(expr, op);
                if (idx < 0) continue;
                var lhs = EvalPrimary(expr.Substring(0, idx).Trim(), ctx);
                var rhs = EvalPrimary(expr.Substring(idx + op.Length).Trim(), ctx);
                return Compare(lhs, rhs, op);
            }

            // Handle ~ (Jinja2 string concatenation)
            int tildeIdx = FindTopLevelOp(expr, "~");
            if (tildeIdx > 0)
            {
                var lhs = EvalPrimary(expr.Substring(0, tildeIdx).Trim(), ctx);
                var rhs = EvalComparison(expr.Substring(tildeIdx + 1).Trim(), ctx);
                return Stringify(lhs) + Stringify(rhs);
            }

            // Handle + (string concatenation / addition)
            int plusIdx = FindTopLevelOp(expr, "+");
            if (plusIdx > 0)
            {
                var lhs = EvalPrimary(expr.Substring(0, plusIdx).Trim(), ctx);
                var rhs = EvalComparison(expr.Substring(plusIdx + 1).Trim(), ctx);
                if (lhs is string ls) return ls + Stringify(rhs);
                if (lhs is int li && rhs is int ri) return li + ri;
                return Stringify(lhs) + Stringify(rhs);
            }

            // Handle - (subtraction)
            int minusIdx = FindTopLevelOp(expr, "-");
            if (minusIdx > 0)
            {
                string leftStr = expr.Substring(0, minusIdx).Trim();
                if (leftStr.Length > 0)
                {
                    var lhs = EvalPrimary(leftStr, ctx);
                    var rhs = EvalPrimary(expr.Substring(minusIdx + 1).Trim(), ctx);
                    return (int)(ToNumber(lhs) - ToNumber(rhs));
                }
            }

            return EvalPrimary(expr, ctx);
        }

        private static object EvalPrimary(string expr, Context ctx)
        {
            expr = expr.Trim();
            if (expr.Length == 0) return null;

            // Parenthesized expression
            if (expr[0] == '(' && FindClosing(expr, 0, '(', ')') == expr.Length - 1)
                return EvalExpr(expr.Substring(1, expr.Length - 2), ctx);

            // Apply filters: expr | filter
            int pipeIdx = FindTopLevelOp(expr, "|");
            if (pipeIdx > 0)
            {
                string baseExpr = expr.Substring(0, pipeIdx).Trim();
                string filter = expr.Substring(pipeIdx + 1).Trim();
                return ApplyFilter(EvalPrimary(baseExpr, ctx), filter, ctx);
            }

            // String literal
            if ((expr[0] == '\'' && expr[expr.Length - 1] == '\'') ||
                (expr[0] == '"' && expr[expr.Length - 1] == '"'))
                return Unescape(expr.Substring(1, expr.Length - 2));

            // Integer literal
            if (int.TryParse(expr, NumberStyles.Integer, CultureInfo.InvariantCulture, out int ival))
                return ival;

            // Float literal
            if (expr.Contains('.') && double.TryParse(expr, NumberStyles.Float, CultureInfo.InvariantCulture, out double dval))
                return dval;

            // Boolean / None literals
            if (expr == "true" || expr == "True") return true;
            if (expr == "false" || expr == "False") return false;
            if (expr == "none" || expr == "None" || expr == "null") return null;

            // List literal [a, b, c]
            if (expr[0] == '[' && expr[expr.Length - 1] == ']')
            {
                string inner = expr.Substring(1, expr.Length - 2).Trim();
                if (inner.Length == 0) return new List<object>();
                var items = SplitArgs(inner);
                var list = new List<object>();
                foreach (var item in items) list.Add(EvalExpr(item, ctx));
                return list;
            }

            // Dict literal {k: v, ...}
            if (expr[0] == '{' && expr[expr.Length - 1] == '}')
            {
                string inner = expr.Substring(1, expr.Length - 2).Trim();
                var dict = new Dictionary<string, object>();
                if (inner.Length > 0)
                {
                    var items = SplitArgs(inner);
                    foreach (var item in items)
                    {
                        int colon = item.IndexOf(':');
                        if (colon > 0)
                        {
                            string k = item.Substring(0, colon).Trim().Trim('\'', '"');
                            string v = item.Substring(colon + 1).Trim();
                            dict[k] = EvalExpr(v, ctx);
                        }
                    }
                }
                return dict;
            }

            // namespace(k=v) constructor
            if (expr.StartsWith("namespace(") && expr.EndsWith(")"))
            {
                string inner = expr.Substring(10, expr.Length - 11);
                var dict = new Dictionary<string, object>();
                var args = SplitArgs(inner);
                foreach (var arg in args)
                {
                    int eq = arg.IndexOf('=');
                    if (eq > 0)
                    {
                        string k = arg.Substring(0, eq).Trim();
                        string v = arg.Substring(eq + 1).Trim();
                        dict[k] = EvalExpr(v, ctx);
                    }
                }
                return dict;
            }

            // raise_exception('...') → just return null (will be handled gracefully)
            if (expr.StartsWith("raise_exception("))
                return null;

            // range(n) function
            if (expr.StartsWith("range(") && expr.EndsWith(")"))
            {
                string inner = expr.Substring(6, expr.Length - 7).Trim();
                int n = (int)ToNumber(EvalExpr(inner, ctx));
                var list = new List<object>();
                for (int j = 0; j < n; j++) list.Add(j);
                return list;
            }

            // strftime_now(format) function
            if (expr.StartsWith("strftime_now(") && expr.EndsWith(")"))
            {
                string inner = expr.Substring(13, expr.Length - 14).Trim();
                string fmt = Stringify(EvalExpr(inner, ctx));
                return ConvertStrftime(fmt);
            }

            // Variable access chain: var.attr['key'][idx].method(args)
            return EvalAccessChain(expr, ctx);
        }

        private static object EvalAccessChain(string expr, Context ctx)
        {
            // Parse the root variable name
            int i = 0;
            int start = 0;

            // Find end of root identifier
            while (i < expr.Length && (char.IsLetterOrDigit(expr[i]) || expr[i] == '_'))
                i++;

            string rootName = expr.Substring(start, i);
            if (rootName.Length == 0) return null;

            object current = ctx.Get(rootName);

            while (i < expr.Length)
            {
                if (expr[i] == '.')
                {
                    i++;
                    int attrStart = i;
                    while (i < expr.Length && (char.IsLetterOrDigit(expr[i]) || expr[i] == '_'))
                        i++;
                    string attr = expr.Substring(attrStart, i - attrStart);

                    // Method call: .method(args)
                    if (i < expr.Length && expr[i] == '(')
                    {
                        int close = FindClosing(expr, i, '(', ')');
                        string argsStr = expr.Substring(i + 1, close - i - 1).Trim();
                        i = close + 1;
                        current = CallMethod(current, attr, argsStr, ctx);
                    }
                    else
                    {
                        current = GetAttr(current, attr);
                    }
                }
                else if (expr[i] == '[')
                {
                    int close = FindClosing(expr, i, '[', ']');
                    string indexExpr = expr.Substring(i + 1, close - i - 1).Trim();
                    i = close + 1;

                    // Array slicing: [1:], [:2], [1:3]
                    int colonIdx = indexExpr.IndexOf(':');
                    if (colonIdx >= 0)
                    {
                        current = SliceCollection(current, indexExpr, ctx);
                    }
                    else
                    {
                        var key = EvalExpr(indexExpr, ctx);
                        current = GetItem(current, key);
                    }
                }
                else if (expr[i] == '(')
                {
                    int close = FindClosing(expr, i, '(', ')');
                    string callArgs = expr.Substring(i + 1, close - i - 1).Trim();
                    i = close + 1;

                    if (current is MacroNode mn)
                        current = CallMacro(mn, callArgs, ctx);
                }
                else
                {
                    break;
                }
            }

            return current;
        }

        #endregion

        #region Helpers

        private static bool EvalIsTest(string lhsExpr, string testName, Context ctx)
        {
            switch (testName)
            {
                case "defined":
                {
                    string root = ExtractRootVar(lhsExpr);
                    if (!ctx.IsDefined(root)) return false;
                    if (lhsExpr.Contains('.') || lhsExpr.Contains('['))
                    {
                        try { return EvalPrimary(lhsExpr, ctx) != null; }
                        catch { return false; }
                    }
                    return true;
                }
                case "undefined":
                {
                    string root = ExtractRootVar(lhsExpr);
                    if (!ctx.IsDefined(root)) return true;
                    if (lhsExpr.Contains('.') || lhsExpr.Contains('['))
                    {
                        try { return EvalPrimary(lhsExpr, ctx) == null; }
                        catch { return true; }
                    }
                    return false;
                }
                case "none":
                    return EvalPrimary(lhsExpr, ctx) == null;
                case "mapping":
                    return EvalPrimary(lhsExpr, ctx) is IDictionary<string, object>;
                case "string":
                    return EvalPrimary(lhsExpr, ctx) is string;
                case "boolean":
                    return EvalPrimary(lhsExpr, ctx) is bool;
                case "sequence":
                {
                    var val = EvalPrimary(lhsExpr, ctx);
                    return val is IList<object> || val is string;
                }
                case "number" or "integer" or "float":
                    return EvalPrimary(lhsExpr, ctx) is int or double;
                case "true":
                    return Truthy(EvalPrimary(lhsExpr, ctx));
                case "false":
                    return !Truthy(EvalPrimary(lhsExpr, ctx));
                case "iterable":
                {
                    var val = EvalPrimary(lhsExpr, ctx);
                    return val is IList<object> || val is IDictionary<string, object> || val is string;
                }
                default:
                    return false;
            }
        }

        private static string Stringify(object val)
        {
            if (val == null) return "";
            if (val is bool b) return b ? "True" : "False";
            if (val is string s) return s;
            return Convert.ToString(val, CultureInfo.InvariantCulture);
        }

        private static string ToJson(object val)
        {
            if (val == null) return "null";
            if (val is bool b) return b ? "true" : "false";
            if (val is string s) return System.Text.Json.JsonSerializer.Serialize(s);
            if (val is int i) return i.ToString(CultureInfo.InvariantCulture);
            if (val is double d) return d.ToString(CultureInfo.InvariantCulture);
            if (val is IList<object> list)
            {
                var sb = new StringBuilder("[");
                for (int j = 0; j < list.Count; j++)
                {
                    if (j > 0) sb.Append(", ");
                    sb.Append(ToJson(list[j]));
                }
                sb.Append(']');
                return sb.ToString();
            }
            if (val is IDictionary<string, object> dict)
            {
                var sb = new StringBuilder("{");
                bool first = true;
                foreach (var kv in dict)
                {
                    if (!first) sb.Append(", ");
                    first = false;
                    sb.Append(System.Text.Json.JsonSerializer.Serialize(kv.Key));
                    sb.Append(": ");
                    sb.Append(ToJson(kv.Value));
                }
                sb.Append('}');
                return sb.ToString();
            }
            return System.Text.Json.JsonSerializer.Serialize(val.ToString());
        }

        private static bool Truthy(object val)
        {
            if (val == null) return false;
            if (val is bool b) return b;
            if (val is string s) return s.Length > 0;
            if (val is int i) return i != 0;
            if (val is double d) return d != 0;
            if (val is IList<object> list) return list.Count > 0;
            if (val is IDictionary<string, object> dict) return dict.Count > 0;
            return true;
        }

        private static double ToNumber(object val)
        {
            if (val is int i) return i;
            if (val is double d) return d;
            if (val is string s && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double r))
                return r;
            return 0;
        }

        private static bool Compare(object lhs, object rhs, string op)
        {
            if (op == "==") return Equals(lhs, rhs);
            if (op == "!=") return !Equals(lhs, rhs);
            double l = ToNumber(lhs), r = ToNumber(rhs);
            return op switch
            {
                "<" => l < r,
                ">" => l > r,
                "<=" => l <= r,
                ">=" => l >= r,
                _ => false
            };
        }

        private static new bool Equals(object a, object b)
        {
            if (a == null && b == null) return true;
            if (a == null || b == null) return false;
            if (a is string sa && b is string sb) return sa == sb;
            if (a is int ia && b is int ib) return ia == ib;
            if (a is bool ba && b is bool bb) return ba == bb;
            return a.ToString() == b.ToString();
        }

        private static bool ContainsValue(object collection, object needle)
        {
            if (collection is string s && needle is string sub)
                return s.Contains(sub);
            if (collection is IList<object> list)
            {
                foreach (var item in list)
                    if (Equals(item, needle)) return true;
                return false;
            }
            if (collection is IDictionary<string, object> dict && needle is string key)
                return dict.ContainsKey(key);
            return false;
        }

        private static object GetAttr(object obj, string attr)
        {
            if (obj is IDictionary<string, object> dict)
                return dict.TryGetValue(attr, out var v) ? v : null;
            return null;
        }

        private static object GetItem(object obj, object key)
        {
            if (obj is IDictionary<string, object> dict && key is string sk)
                return dict.TryGetValue(sk, out var v) ? v : null;
            if (obj is IList<object> list && key is int idx)
            {
                if (idx < 0) idx += list.Count;
                return (idx >= 0 && idx < list.Count) ? list[idx] : null;
            }
            if (obj is string s && key is int si)
            {
                if (si < 0) si += s.Length;
                return (si >= 0 && si < s.Length) ? s[si].ToString() : null;
            }
            return null;
        }

        private static object SliceCollection(object obj, string sliceExpr, Context ctx)
        {
            // Handle step-based slicing like [::-1]
            var colonParts = sliceExpr.Split(':');
            string startStr = colonParts[0].Trim();
            string endStr = colonParts.Length > 1 ? colonParts[1].Trim() : "";
            string stepStr = colonParts.Length > 2 ? colonParts[2].Trim() : "";

            int step = stepStr.Length > 0 ? (int)ToNumber(EvalExpr(stepStr, ctx)) : 1;

            if (obj is IList<object> list)
            {
                int len = list.Count;
                if (step == -1 && startStr.Length == 0 && endStr.Length == 0)
                {
                    var reversed = new List<object>(list);
                    reversed.Reverse();
                    return reversed;
                }
                int s = startStr.Length > 0 ? (int)ToNumber(EvalExpr(startStr, ctx)) : 0;
                int e = endStr.Length > 0 ? (int)ToNumber(EvalExpr(endStr, ctx)) : len;
                if (s < 0) s += len;
                if (e < 0) e += len;
                s = Math.Max(0, Math.Min(s, len));
                e = Math.Max(s, Math.Min(e, len));
                var result = new List<object>();
                for (int i = s; i < e; i++) result.Add(list[i]);
                return result;
            }

            if (obj is string str)
            {
                int len = str.Length;
                if (step == -1 && startStr.Length == 0 && endStr.Length == 0)
                {
                    char[] chars = str.ToCharArray();
                    Array.Reverse(chars);
                    return new string(chars);
                }
                int s = startStr.Length > 0 ? (int)ToNumber(EvalExpr(startStr, ctx)) : 0;
                int e = endStr.Length > 0 ? (int)ToNumber(EvalExpr(endStr, ctx)) : len;
                if (s < 0) s += len;
                if (e < 0) e += len;
                s = Math.Max(0, Math.Min(s, len));
                e = Math.Max(s, Math.Min(e, len));
                return str.Substring(s, e - s);
            }

            return null;
        }

        private static object CallMethod(object obj, string method, string argsStr, Context ctx)
        {
            switch (method)
            {
                case "get":
                {
                    var args = SplitArgs(argsStr);
                    if (args.Count == 0) return null;
                    var key = EvalExpr(args[0], ctx);
                    object result = GetItem(obj, key);
                    if (result == null && args.Count > 1)
                        result = EvalExpr(args[1], ctx);
                    return result;
                }
                case "items":
                {
                    if (obj is IDictionary<string, object> dict)
                    {
                        var list = new List<object>();
                        foreach (var kv in dict)
                            list.Add(new KeyValuePair<string, object>(kv.Key, kv.Value));
                        return list;
                    }
                    return new List<object>();
                }
                case "strip" or "trim":
                    return obj is string s ? s.Trim() : obj;
                case "split":
                {
                    if (obj is string ss)
                    {
                        var parts = SplitArgs(argsStr);
                        string sep2 = parts.Count > 0 ? Stringify(EvalExpr(parts[0], ctx)) : " ";
                        var list = new List<object>();
                        foreach (string p in ss.Split(new[] { sep2 }, StringSplitOptions.None))
                            list.Add(p);
                        return list;
                    }
                    return obj;
                }
                case "upper":
                    return obj is string su ? su.ToUpperInvariant() : obj;
                case "lower":
                    return obj is string sl ? sl.ToLowerInvariant() : obj;
                case "startswith":
                {
                    var args = SplitArgs(argsStr);
                    if (args.Count > 0 && obj is string ss)
                        return ss.StartsWith(Stringify(EvalExpr(args[0], ctx)));
                    return false;
                }
                case "endswith":
                {
                    var args = SplitArgs(argsStr);
                    if (args.Count > 0 && obj is string se)
                        return se.EndsWith(Stringify(EvalExpr(args[0], ctx)));
                    return false;
                }
                case "append":
                {
                    if (obj is IList<object> list)
                    {
                        var args = SplitArgs(argsStr);
                        if (args.Count > 0) list.Add(EvalExpr(args[0], ctx));
                    }
                    return obj;
                }
                default:
                    return null;
            }
        }

        private static object ApplyFilter(object val, string filter, Context ctx)
        {
            // filter might have args: default('value')
            string filterName = filter;
            string filterArgs = null;
            int paren = filter.IndexOf('(');
            if (paren > 0)
            {
                filterName = filter.Substring(0, paren).Trim();
                filterArgs = filter.Substring(paren + 1, filter.Length - paren - 2).Trim();
            }

            switch (filterName)
            {
                case "trim" or "strip":
                    return val is string s ? s.Trim() : val;
                case "length":
                    if (val is string sl) return sl.Length;
                    if (val is IList<object> list) return list.Count;
                    if (val is IDictionary<string, object> dict) return dict.Count;
                    return 0;
                case "upper":
                    return val is string su ? su.ToUpperInvariant() : val;
                case "lower":
                    return val is string slo ? slo.ToLowerInvariant() : val;
                case "first":
                    if (val is IList<object> fl && fl.Count > 0) return fl[0];
                    return null;
                case "last":
                    if (val is IList<object> ll && ll.Count > 0) return ll[ll.Count - 1];
                    return null;
                case "default" or "d":
                    if (val == null || (val is string ds && ds.Length == 0))
                        return filterArgs != null ? EvalExpr(filterArgs, ctx) : null;
                    return val;
                case "list":
                    return ToList(val);
                case "int":
                    return (int)ToNumber(val);
                case "string":
                    return Stringify(val);
                case "join":
                {
                    string sep = filterArgs != null ? Stringify(EvalExpr(filterArgs, ctx)) : "";
                    if (val is IList<object> jl)
                    {
                        var sb2 = new StringBuilder();
                        for (int j = 0; j < jl.Count; j++)
                        {
                            if (j > 0) sb2.Append(sep);
                            sb2.Append(Stringify(jl[j]));
                        }
                        return sb2.ToString();
                    }
                    return Stringify(val);
                }
                case "tojson" or "tojson(indent=2)" or "tojson()":
                    return ToJson(val);
                case "dictsort":
                {
                    if (val is IDictionary<string, object> dictSort)
                    {
                        var sorted = new List<KeyValuePair<string, object>>(dictSort);
                        sorted.Sort((a, b) => string.Compare(a.Key, b.Key, StringComparison.Ordinal));
                        var result = new List<object>();
                        foreach (var kv in sorted)
                            result.Add(kv);
                        return result;
                    }
                    return val;
                }
                case "safe":
                    return val;
                case "selectattr" or "map" or "reject" or "rejectattr":
                    return val;
                default:
                    return val;
            }
        }

        private static List<object> ToList(object val)
        {
            if (val is IList<object> list) return new List<object>(list);
            if (val is IDictionary<string, object> dict)
            {
                var result = new List<object>();
                foreach (var kv in dict)
                    result.Add(new KeyValuePair<string, object>(kv.Key, kv.Value));
                return result;
            }
            if (val is string s)
            {
                var result = new List<object>();
                foreach (char c in s) result.Add(c.ToString());
                return result;
            }
            return new List<object>();
        }

        private static string ExtractRootVar(string expr)
        {
            int dot = expr.IndexOf('.');
            int bracket = expr.IndexOf('[');
            int end = expr.Length;
            if (dot > 0) end = Math.Min(end, dot);
            if (bracket > 0) end = Math.Min(end, bracket);
            return expr.Substring(0, end).Trim();
        }

        private static string ConvertStrftime(string fmt)
        {
            var now = DateTime.Now;
            var sb = new StringBuilder();
            for (int i = 0; i < fmt.Length; i++)
            {
                if (fmt[i] == '%' && i + 1 < fmt.Length)
                {
                    i++;
                    sb.Append(fmt[i] switch
                    {
                        'Y' => now.ToString("yyyy"),
                        'm' => now.ToString("MM"),
                        'd' => now.ToString("dd"),
                        'H' => now.ToString("HH"),
                        'M' => now.ToString("mm"),
                        'S' => now.ToString("ss"),
                        'A' => now.ToString("dddd"),
                        'B' => now.ToString("MMMM"),
                        'b' => now.ToString("MMM"),
                        'a' => now.ToString("ddd"),
                        'p' => now.ToString("tt"),
                        'I' => now.ToString("hh"),
                        'j' => now.DayOfYear.ToString("D3"),
                        '%' => "%",
                        _ => "%" + fmt[i],
                    });
                }
                else
                {
                    sb.Append(fmt[i]);
                }
            }
            return sb.ToString();
        }

        private static string Unescape(string s)
        {
            if (!s.Contains('\\')) return s;
            var sb = new StringBuilder(s.Length);
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '\\' && i + 1 < s.Length)
                {
                    i++;
                    switch (s[i])
                    {
                        case 'n': sb.Append('\n'); break;
                        case 'r': sb.Append('\r'); break;
                        case 't': sb.Append('\t'); break;
                        case '\\': sb.Append('\\'); break;
                        case '\'': sb.Append('\''); break;
                        case '"': sb.Append('"'); break;
                        default: sb.Append('\\'); sb.Append(s[i]); break;
                    }
                }
                else
                {
                    sb.Append(s[i]);
                }
            }
            return sb.ToString();
        }

        private static int FindClosing(string s, int openPos, char open, char close)
        {
            int depth = 1;
            bool inSingle = false, inDouble = false;
            for (int i = openPos + 1; i < s.Length; i++)
            {
                char c = s[i];
                if (c == '\'' && !inDouble) inSingle = !inSingle;
                else if (c == '"' && !inSingle) inDouble = !inDouble;
                else if (!inSingle && !inDouble)
                {
                    if (c == open) depth++;
                    else if (c == close && --depth == 0) return i;
                }
            }
            return s.Length - 1;
        }

        private static int FindTopLevelKeyword(string s, string kw)
        {
            int depth = 0;
            bool inSingle = false, inDouble = false;
            for (int i = 0; i <= s.Length - kw.Length; i++)
            {
                char c = s[i];
                if (c == '\'' && !inDouble) { inSingle = !inSingle; continue; }
                if (c == '"' && !inSingle) { inDouble = !inDouble; continue; }
                if (inSingle || inDouble) continue;
                if (c == '(' || c == '[' || c == '{') { depth++; continue; }
                if (c == ')' || c == ']' || c == '}') { depth--; continue; }
                if (depth > 0) continue;
                if (s.Substring(i, kw.Length) == kw) return i;
            }
            return -1;
        }

        private static int FindTopLevelOp(string s, string op)
        {
            int depth = 0;
            bool inSingle = false, inDouble = false;
            for (int i = 0; i <= s.Length - op.Length; i++)
            {
                char c = s[i];
                if (c == '\'' && !inDouble) { inSingle = !inSingle; continue; }
                if (c == '"' && !inSingle) { inDouble = !inDouble; continue; }
                if (inSingle || inDouble) continue;
                if (c == '(' || c == '[' || c == '{') { depth++; continue; }
                if (c == ')' || c == ']' || c == '}') { depth--; continue; }
                if (depth > 0) continue;
                if (s.Substring(i, op.Length) == op)
                    return i;
            }
            return -1;
        }

        private static List<string> SplitTopLevel(string s, string sep)
        {
            var result = new List<string>();
            int start = 0;
            while (true)
            {
                int idx = FindTopLevelKeyword(s.Substring(start), sep);
                if (idx < 0)
                {
                    result.Add(s.Substring(start).Trim());
                    break;
                }
                result.Add(s.Substring(start, idx).Trim());
                start += idx + sep.Length;
            }
            return result;
        }

        private static List<string> SplitArgs(string s)
        {
            var result = new List<string>();
            int start = 0;
            int depth = 0;
            bool inSingle = false, inDouble = false;

            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (c == '\'' && !inDouble) inSingle = !inSingle;
                else if (c == '"' && !inSingle) inDouble = !inDouble;
                else if (!inSingle && !inDouble)
                {
                    if (c == '(' || c == '[' || c == '{') depth++;
                    else if (c == ')' || c == ']' || c == '}') depth--;
                    else if (c == ',' && depth == 0)
                    {
                        result.Add(s.Substring(start, i - start).Trim());
                        start = i + 1;
                    }
                }
            }

            string last = s.Substring(start).Trim();
            if (last.Length > 0) result.Add(last);
            return result;
        }

        #endregion
    }
}
