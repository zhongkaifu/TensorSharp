using System;

namespace AdvUtils
{
    public static class Logger
    {
        public enum Level { debug, info, warn, err }

        public static void WriteLine(string msg) => Console.WriteLine(msg);
        public static void WriteLine(Level level, string msg) => Console.WriteLine($"[{level}] {msg}");
        public static void WriteLine(Level level, ConsoleColor color, string msg) => Console.WriteLine($"[{level}] {msg}");
    }
}
