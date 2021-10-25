/*     */ package org.gradle.cli;
/*     */ 
/*     */ import java.io.OutputStreamWriter;
/*     */ import java.io.PrintWriter;
/*     */ import java.io.Writer;
/*     */ import java.util.ArrayList;
/*     */ import java.util.Arrays;
/*     */ import java.util.Collection;
/*     */ import java.util.Collections;
/*     */ import java.util.Comparator;
/*     */ import java.util.Formatter;
/*     */ import java.util.HashMap;
/*     */ import java.util.HashSet;
/*     */ import java.util.LinkedHashMap;
/*     */ import java.util.List;
/*     */ import java.util.Map;
/*     */ import java.util.Set;
/*     */ import java.util.TreeSet;
/*     */ import java.util.regex.Pattern;
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ public class CommandLineParser
/*     */ {
/*  55 */   private static final Pattern OPTION_NAME_PATTERN = Pattern.compile("(\\?|\\p{Alnum}[\\p{Alnum}-_]*)");
/*     */   
/*  57 */   private Map<String, CommandLineOption> optionsByString = new HashMap<String, CommandLineOption>();
/*     */   private boolean allowMixedOptions;
/*     */   private boolean allowUnknownOptions;
/*     */   private final PrintWriter deprecationPrinter;
/*     */   
/*     */   public CommandLineParser() {
/*  63 */     this(new OutputStreamWriter(System.out));
/*     */   }
/*     */   
/*     */   public CommandLineParser(Writer deprecationPrinter) {
/*  67 */     this.deprecationPrinter = new PrintWriter(deprecationPrinter);
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public ParsedCommandLine parse(String... commandLine) throws CommandLineArgumentException {
/*  79 */     return parse(Arrays.asList(commandLine));
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public ParsedCommandLine parse(Iterable<String> commandLine) throws CommandLineArgumentException {
/*  91 */     ParsedCommandLine parsedCommandLine = new ParsedCommandLine(new HashSet<CommandLineOption>(this.optionsByString.values()));
/*  92 */     ParserState parseState = new BeforeFirstSubCommand(parsedCommandLine);
/*  93 */     for (String arg : commandLine) {
/*  94 */       if (parseState.maybeStartOption(arg)) {
/*  95 */         if (arg.equals("--")) {
/*  96 */           parseState = new AfterOptions(parsedCommandLine); continue;
/*  97 */         }  if (arg.matches("--[^=]+")) {
/*  98 */           OptionParserState optionParserState = parseState.onStartOption(arg, arg.substring(2));
/*  99 */           parseState = optionParserState.onStartNextArg(); continue;
/* 100 */         }  if (arg.matches("(?s)--[^=]+=.*")) {
/* 101 */           int endArg = arg.indexOf('=');
/* 102 */           OptionParserState optionParserState = parseState.onStartOption(arg, arg.substring(2, endArg));
/* 103 */           parseState = optionParserState.onArgument(arg.substring(endArg + 1)); continue;
/* 104 */         }  if (arg.matches("(?s)-[^=]=.*")) {
/* 105 */           OptionParserState optionParserState = parseState.onStartOption(arg, arg.substring(1, 2));
/* 106 */           parseState = optionParserState.onArgument(arg.substring(3)); continue;
/*     */         } 
/* 108 */         assert arg.matches("(?s)-[^-].*");
/* 109 */         String option = arg.substring(1);
/* 110 */         if (this.optionsByString.containsKey(option)) {
/* 111 */           OptionParserState optionParserState = parseState.onStartOption(arg, option);
/* 112 */           parseState = optionParserState.onStartNextArg(); continue;
/*     */         } 
/* 114 */         String option1 = arg.substring(1, 2);
/*     */         
/* 116 */         if (this.optionsByString.containsKey(option1)) {
/* 117 */           OptionParserState optionParserState = parseState.onStartOption("-" + option1, option1);
/* 118 */           if (optionParserState.getHasArgument()) {
/* 119 */             parseState = optionParserState.onArgument(arg.substring(2)); continue;
/*     */           } 
/* 121 */           parseState = optionParserState.onComplete();
/* 122 */           for (int i = 2; i < arg.length(); i++) {
/* 123 */             String optionStr = arg.substring(i, i + 1);
/* 124 */             optionParserState = parseState.onStartOption("-" + optionStr, optionStr);
/* 125 */             parseState = optionParserState.onComplete();
/*     */           } 
/*     */           continue;
/*     */         } 
/* 129 */         if (this.allowUnknownOptions) {
/*     */           
/* 131 */           OptionParserState optionParserState = parseState.onStartOption(arg, option);
/* 132 */           parseState = optionParserState.onComplete();
/*     */           
/*     */           continue;
/*     */         } 
/* 136 */         OptionParserState parsedOption = parseState.onStartOption("-" + option1, option1);
/* 137 */         parseState = parsedOption.onComplete();
/*     */ 
/*     */         
/*     */         continue;
/*     */       } 
/*     */       
/* 143 */       parseState = parseState.onNonOption(arg);
/*     */     } 
/*     */ 
/*     */     
/* 147 */     parseState.onCommandLineEnd();
/* 148 */     return parsedCommandLine;
/*     */   }
/*     */   
/*     */   public CommandLineParser allowMixedSubcommandsAndOptions() {
/* 152 */     this.allowMixedOptions = true;
/* 153 */     return this;
/*     */   }
/*     */   
/*     */   public CommandLineParser allowUnknownOptions() {
/* 157 */     this.allowUnknownOptions = true;
/* 158 */     return this;
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public CommandLineParser allowOneOf(String... options) {
/* 166 */     Set<CommandLineOption> commandLineOptions = new HashSet<CommandLineOption>();
/* 167 */     for (String option : options) {
/* 168 */       commandLineOptions.add(this.optionsByString.get(option));
/*     */     }
/* 170 */     for (CommandLineOption commandLineOption : commandLineOptions) {
/* 171 */       commandLineOption.groupWith(commandLineOptions);
/*     */     }
/* 173 */     return this;
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public void printUsage(Appendable out) {
/* 182 */     Formatter formatter = new Formatter(out);
/* 183 */     Set<CommandLineOption> orderedOptions = new TreeSet<CommandLineOption>(new OptionComparator());
/* 184 */     orderedOptions.addAll(this.optionsByString.values());
/* 185 */     Map<String, String> lines = new LinkedHashMap<String, String>();
/* 186 */     for (CommandLineOption option : orderedOptions) {
/* 187 */       Set<String> orderedOptionStrings = new TreeSet<String>(new OptionStringComparator());
/* 188 */       orderedOptionStrings.addAll(option.getOptions());
/* 189 */       List<String> prefixedStrings = new ArrayList<String>();
/* 190 */       for (String optionString : orderedOptionStrings) {
/* 191 */         if (optionString.length() == 1) {
/* 192 */           prefixedStrings.add("-" + optionString); continue;
/*     */         } 
/* 194 */         prefixedStrings.add("--" + optionString);
/*     */       } 
/*     */ 
/*     */       
/* 198 */       String key = join(prefixedStrings, ", ");
/* 199 */       String value = option.getDescription();
/* 200 */       if (value == null || value.length() == 0) {
/* 201 */         value = "";
/*     */       }
/*     */       
/* 204 */       lines.put(key, value);
/*     */     } 
/* 206 */     int max = 0;
/* 207 */     for (String optionStr : lines.keySet()) {
/* 208 */       max = Math.max(max, optionStr.length());
/*     */     }
/* 210 */     for (Map.Entry<String, String> entry : lines.entrySet()) {
/* 211 */       if (((String)entry.getValue()).length() == 0) {
/* 212 */         formatter.format("%s%n", new Object[] { entry.getKey() }); continue;
/*     */       } 
/* 214 */       formatter.format("%-" + max + "s  %s%n", new Object[] { entry.getKey(), entry.getValue() });
/*     */     } 
/*     */     
/* 217 */     formatter.flush();
/*     */   }
/*     */   
/*     */   private static String join(Collection<?> things, String separator) {
/* 221 */     StringBuffer buffer = new StringBuffer();
/* 222 */     boolean first = true;
/*     */     
/* 224 */     if (separator == null) {
/* 225 */       separator = "";
/*     */     }
/*     */     
/* 228 */     for (Object thing : things) {
/* 229 */       if (!first) {
/* 230 */         buffer.append(separator);
/*     */       }
/* 232 */       buffer.append(thing.toString());
/* 233 */       first = false;
/*     */     } 
/* 235 */     return buffer.toString();
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public CommandLineOption option(String... options) {
/* 245 */     for (String str : options) {
/* 246 */       if (this.optionsByString.containsKey(str)) {
/* 247 */         throw new IllegalArgumentException(String.format("Option '%s' is already defined.", new Object[] { str }));
/*     */       }
/* 249 */       if (str.startsWith("-")) {
/* 250 */         throw new IllegalArgumentException(String.format("Cannot add option '%s' as an option cannot start with '-'.", new Object[] { str }));
/*     */       }
/* 252 */       if (!OPTION_NAME_PATTERN.matcher(str).matches()) {
/* 253 */         throw new IllegalArgumentException(String.format("Cannot add option '%s' as an option can only contain alphanumeric characters or '-' or '_'.", new Object[] { str }));
/*     */       }
/*     */     } 
/* 256 */     CommandLineOption option = new CommandLineOption(Arrays.asList(options));
/* 257 */     for (String optionStr : option.getOptions()) {
/* 258 */       this.optionsByString.put(optionStr, option);
/*     */     }
/* 260 */     return option;
/*     */   }
/*     */   
/*     */   private static class OptionString {
/*     */     private final String arg;
/*     */     private final String option;
/*     */     
/*     */     private OptionString(String arg, String option) {
/* 268 */       this.arg = arg;
/* 269 */       this.option = option;
/*     */     }
/*     */     
/*     */     public String getDisplayName() {
/* 273 */       return this.arg.startsWith("--") ? ("--" + this.option) : ("-" + this.option);
/*     */     }
/*     */ 
/*     */     
/*     */     public String toString() {
/* 278 */       return getDisplayName();
/*     */     }
/*     */   }
/*     */   
/*     */   private static abstract class ParserState {
/*     */     private ParserState() {}
/*     */     
/*     */     boolean isOption(String arg) {
/* 286 */       return arg.matches("(?s)-.+");
/*     */     }
/*     */     
/*     */     public void onCommandLineEnd() {}
/*     */     
/*     */     public abstract boolean maybeStartOption(String param1String);
/*     */     
/*     */     public abstract CommandLineParser.OptionParserState onStartOption(String param1String1, String param1String2);
/*     */     
/*     */     public abstract ParserState onNonOption(String param1String); }
/*     */   
/*     */   private abstract class OptionAwareParserState extends ParserState {
/*     */     protected final ParsedCommandLine commandLine;
/*     */     
/*     */     protected OptionAwareParserState(ParsedCommandLine commandLine) {
/* 301 */       this.commandLine = commandLine;
/*     */     }
/*     */ 
/*     */     
/*     */     public boolean maybeStartOption(String arg) {
/* 306 */       return isOption(arg);
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onNonOption(String arg) {
/* 311 */       this.commandLine.addExtraValue(arg);
/* 312 */       return CommandLineParser.this.allowMixedOptions ? new CommandLineParser.AfterFirstSubCommand(this.commandLine) : new CommandLineParser.AfterOptions(this.commandLine);
/*     */     }
/*     */   }
/*     */   
/*     */   private class BeforeFirstSubCommand extends OptionAwareParserState {
/*     */     private BeforeFirstSubCommand(ParsedCommandLine commandLine) {
/* 318 */       super(commandLine);
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.OptionParserState onStartOption(String arg, String option) {
/* 323 */       CommandLineParser.OptionString optionString = new CommandLineParser.OptionString(arg, option);
/* 324 */       CommandLineOption commandLineOption = (CommandLineOption)CommandLineParser.this.optionsByString.get(option);
/* 325 */       if (commandLineOption == null) {
/* 326 */         if (CommandLineParser.this.allowUnknownOptions) {
/* 327 */           return new CommandLineParser.UnknownOptionParserState(arg, this.commandLine, this);
/*     */         }
/* 329 */         throw new CommandLineArgumentException(String.format("Unknown command-line option '%s'.", new Object[] { optionString }));
/*     */       } 
/*     */       
/* 332 */       return new CommandLineParser.KnownOptionParserState(optionString, commandLineOption, this.commandLine, this);
/*     */     }
/*     */   }
/*     */   
/*     */   private class AfterFirstSubCommand extends OptionAwareParserState {
/*     */     private AfterFirstSubCommand(ParsedCommandLine commandLine) {
/* 338 */       super(commandLine);
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.OptionParserState onStartOption(String arg, String option) {
/* 343 */       CommandLineOption commandLineOption = (CommandLineOption)CommandLineParser.this.optionsByString.get(option);
/* 344 */       if (commandLineOption == null) {
/* 345 */         return new CommandLineParser.UnknownOptionParserState(arg, this.commandLine, this);
/*     */       }
/* 347 */       return new CommandLineParser.KnownOptionParserState(new CommandLineParser.OptionString(arg, option), commandLineOption, this.commandLine, this);
/*     */     }
/*     */   }
/*     */   
/*     */   private static class AfterOptions extends ParserState {
/*     */     private final ParsedCommandLine commandLine;
/*     */     
/*     */     private AfterOptions(ParsedCommandLine commandLine) {
/* 355 */       this.commandLine = commandLine;
/*     */     }
/*     */ 
/*     */     
/*     */     public boolean maybeStartOption(String arg) {
/* 360 */       return false;
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.OptionParserState onStartOption(String arg, String option) {
/* 365 */       return new CommandLineParser.UnknownOptionParserState(arg, this.commandLine, this);
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onNonOption(String arg) {
/* 370 */       this.commandLine.addExtraValue(arg);
/* 371 */       return this;
/*     */     }
/*     */   }
/*     */   
/*     */   private static class MissingOptionArgState extends ParserState {
/*     */     private final CommandLineParser.OptionParserState option;
/*     */     
/*     */     private MissingOptionArgState(CommandLineParser.OptionParserState option) {
/* 379 */       this.option = option;
/*     */     }
/*     */ 
/*     */     
/*     */     public boolean maybeStartOption(String arg) {
/* 384 */       return isOption(arg);
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.OptionParserState onStartOption(String arg, String option) {
/* 389 */       return this.option.onComplete().onStartOption(arg, option);
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onNonOption(String arg) {
/* 394 */       return this.option.onArgument(arg);
/*     */     }
/*     */ 
/*     */     
/*     */     public void onCommandLineEnd() {
/* 399 */       this.option.onComplete();
/*     */     }
/*     */   }
/*     */   
/*     */   private static abstract class OptionParserState { private OptionParserState() {}
/*     */     
/*     */     public abstract CommandLineParser.ParserState onStartNextArg();
/*     */     
/*     */     public abstract CommandLineParser.ParserState onArgument(String param1String);
/*     */     
/*     */     public abstract boolean getHasArgument();
/*     */     
/*     */     public abstract CommandLineParser.ParserState onComplete(); }
/*     */   
/*     */   private class KnownOptionParserState extends OptionParserState {
/*     */     private final CommandLineParser.OptionString optionString;
/*     */     private final CommandLineOption option;
/*     */     private final ParsedCommandLine commandLine;
/*     */     private final CommandLineParser.ParserState state;
/* 418 */     private final List<String> values = new ArrayList<String>();
/*     */     
/*     */     private KnownOptionParserState(CommandLineParser.OptionString optionString, CommandLineOption option, ParsedCommandLine commandLine, CommandLineParser.ParserState state) {
/* 421 */       this.optionString = optionString;
/* 422 */       this.option = option;
/* 423 */       this.commandLine = commandLine;
/* 424 */       this.state = state;
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onArgument(String argument) {
/* 429 */       if (!getHasArgument()) {
/* 430 */         throw new CommandLineArgumentException(String.format("Command-line option '%s' does not take an argument.", new Object[] { this.optionString }));
/*     */       }
/* 432 */       if (argument.length() == 0) {
/* 433 */         throw new CommandLineArgumentException(String.format("An empty argument was provided for command-line option '%s'.", new Object[] { this.optionString }));
/*     */       }
/* 435 */       this.values.add(argument);
/* 436 */       return onComplete();
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onStartNextArg() {
/* 441 */       if (this.option.getAllowsArguments() && this.values.isEmpty()) {
/* 442 */         return new CommandLineParser.MissingOptionArgState(this);
/*     */       }
/* 444 */       return onComplete();
/*     */     }
/*     */ 
/*     */     
/*     */     public boolean getHasArgument() {
/* 449 */       return this.option.getAllowsArguments();
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onComplete() {
/* 454 */       if (getHasArgument() && this.values.isEmpty()) {
/* 455 */         throw new CommandLineArgumentException(String.format("No argument was provided for command-line option '%s'.", new Object[] { this.optionString }));
/*     */       }
/*     */       
/* 458 */       ParsedCommandLineOption parsedOption = this.commandLine.addOption(this.optionString.option, this.option);
/* 459 */       if (this.values.size() + parsedOption.getValues().size() > 1 && !this.option.getAllowsMultipleArguments()) {
/* 460 */         throw new CommandLineArgumentException(String.format("Multiple arguments were provided for command-line option '%s'.", new Object[] { this.optionString }));
/*     */       }
/* 462 */       for (String value : this.values) {
/* 463 */         parsedOption.addArgument(value);
/*     */       }
/* 465 */       if (this.option.getDeprecationWarning() != null) {
/* 466 */         CommandLineParser.this.deprecationPrinter.println("The " + this.optionString + " option is deprecated - " + this.option.getDeprecationWarning());
/*     */       }
/*     */       
/* 469 */       for (CommandLineOption otherOption : this.option.getGroupWith()) {
/* 470 */         this.commandLine.removeOption(otherOption);
/*     */       }
/*     */       
/* 473 */       return this.state;
/*     */     }
/*     */   }
/*     */   
/*     */   private static class UnknownOptionParserState extends OptionParserState {
/*     */     private final CommandLineParser.ParserState state;
/*     */     private final String arg;
/*     */     private final ParsedCommandLine commandLine;
/*     */     
/*     */     private UnknownOptionParserState(String arg, ParsedCommandLine commandLine, CommandLineParser.ParserState state) {
/* 483 */       this.arg = arg;
/* 484 */       this.commandLine = commandLine;
/* 485 */       this.state = state;
/*     */     }
/*     */ 
/*     */     
/*     */     public boolean getHasArgument() {
/* 490 */       return true;
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onStartNextArg() {
/* 495 */       return onComplete();
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onArgument(String argument) {
/* 500 */       return onComplete();
/*     */     }
/*     */ 
/*     */     
/*     */     public CommandLineParser.ParserState onComplete() {
/* 505 */       this.commandLine.addExtraValue(this.arg);
/* 506 */       return this.state;
/*     */     } }
/*     */   
/*     */   private static final class OptionComparator implements Comparator<CommandLineOption> { private OptionComparator() {}
/*     */     
/*     */     public int compare(CommandLineOption option1, CommandLineOption option2) {
/* 512 */       String min1 = Collections.<String>min(option1.getOptions(), new CommandLineParser.OptionStringComparator());
/* 513 */       String min2 = Collections.<String>min(option2.getOptions(), new CommandLineParser.OptionStringComparator());
/* 514 */       return (new CommandLineParser.CaseInsensitiveStringComparator()).compare(min1, min2);
/*     */     } }
/*     */   
/*     */   private static final class CaseInsensitiveStringComparator implements Comparator<String> { private CaseInsensitiveStringComparator() {}
/*     */     
/*     */     public int compare(String option1, String option2) {
/* 520 */       int diff = option1.compareToIgnoreCase(option2);
/* 521 */       if (diff != 0) {
/* 522 */         return diff;
/*     */       }
/* 524 */       return option1.compareTo(option2);
/*     */     } }
/*     */   
/*     */   private static final class OptionStringComparator implements Comparator<String> { private OptionStringComparator() {}
/*     */     
/*     */     public int compare(String option1, String option2) {
/* 530 */       boolean short1 = (option1.length() == 1);
/* 531 */       boolean short2 = (option2.length() == 1);
/* 532 */       if (short1 && !short2) {
/* 533 */         return -1;
/*     */       }
/* 535 */       if (!short1 && short2) {
/* 536 */         return 1;
/*     */       }
/* 538 */       return (new CommandLineParser.CaseInsensitiveStringComparator()).compare(option1, option2);
/*     */     } }
/*     */ 
/*     */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\CommandLineParser.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */