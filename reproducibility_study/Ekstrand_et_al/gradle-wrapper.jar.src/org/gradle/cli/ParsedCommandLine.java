/*     */ package org.gradle.cli;
/*     */ 
/*     */ import java.util.ArrayList;
/*     */ import java.util.Collection;
/*     */ import java.util.HashMap;
/*     */ import java.util.HashSet;
/*     */ import java.util.List;
/*     */ import java.util.Map;
/*     */ import java.util.Set;
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ public class ParsedCommandLine
/*     */ {
/*  21 */   private final Map<String, ParsedCommandLineOption> optionsByString = new HashMap<String, ParsedCommandLineOption>();
/*  22 */   private final Set<String> presentOptions = new HashSet<String>();
/*  23 */   private final List<String> extraArguments = new ArrayList<String>();
/*     */   
/*     */   ParsedCommandLine(Iterable<CommandLineOption> options) {
/*  26 */     for (CommandLineOption option : options) {
/*  27 */       ParsedCommandLineOption parsedOption = new ParsedCommandLineOption();
/*  28 */       for (String optionStr : option.getOptions()) {
/*  29 */         this.optionsByString.put(optionStr, parsedOption);
/*     */       }
/*     */     } 
/*     */   }
/*     */ 
/*     */   
/*     */   public String toString() {
/*  36 */     return String.format("options: %s, extraArguments: %s", new Object[] { quoteAndJoin(this.presentOptions), quoteAndJoin(this.extraArguments) });
/*     */   }
/*     */   
/*     */   private String quoteAndJoin(Iterable<String> strings) {
/*  40 */     StringBuilder output = new StringBuilder();
/*  41 */     boolean isFirst = true;
/*  42 */     for (String string : strings) {
/*  43 */       if (!isFirst) {
/*  44 */         output.append(", ");
/*     */       }
/*  46 */       output.append("'");
/*  47 */       output.append(string);
/*  48 */       output.append("'");
/*  49 */       isFirst = false;
/*     */     } 
/*  51 */     return output.toString();
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public boolean hasOption(String option) {
/*  61 */     option(option);
/*  62 */     return this.presentOptions.contains(option);
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public boolean hasAnyOption(Collection<String> logLevelOptions) {
/*  72 */     for (String option : logLevelOptions) {
/*  73 */       if (hasOption(option)) {
/*  74 */         return true;
/*     */       }
/*     */     } 
/*  77 */     return false;
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public ParsedCommandLineOption option(String option) {
/*  87 */     ParsedCommandLineOption parsedOption = this.optionsByString.get(option);
/*  88 */     if (parsedOption == null) {
/*  89 */       throw new IllegalArgumentException(String.format("Option '%s' not defined.", new Object[] { option }));
/*     */     }
/*  91 */     return parsedOption;
/*     */   }
/*     */   
/*     */   public List<String> getExtraArguments() {
/*  95 */     return this.extraArguments;
/*     */   }
/*     */   
/*     */   void addExtraValue(String value) {
/*  99 */     this.extraArguments.add(value);
/*     */   }
/*     */   
/*     */   ParsedCommandLineOption addOption(String optionStr, CommandLineOption option) {
/* 103 */     ParsedCommandLineOption parsedOption = this.optionsByString.get(optionStr);
/* 104 */     this.presentOptions.addAll(option.getOptions());
/* 105 */     return parsedOption;
/*     */   }
/*     */   
/*     */   void removeOption(CommandLineOption option) {
/* 109 */     this.presentOptions.removeAll(option.getOptions());
/*     */   }
/*     */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\ParsedCommandLine.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */