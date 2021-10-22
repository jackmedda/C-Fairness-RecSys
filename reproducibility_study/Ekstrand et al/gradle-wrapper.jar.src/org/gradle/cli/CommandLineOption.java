/*     */ package org.gradle.cli;
/*     */ 
/*     */ import java.util.HashSet;
/*     */ import java.util.List;
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
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ public class CommandLineOption
/*     */ {
/*  23 */   private final Set<String> options = new HashSet<String>();
/*  24 */   private Class<?> argumentType = void.class;
/*     */   private String description;
/*     */   private String deprecationWarning;
/*     */   private boolean incubating;
/*  28 */   private final Set<CommandLineOption> groupWith = new HashSet<CommandLineOption>();
/*     */   
/*     */   public CommandLineOption(Iterable<String> options) {
/*  31 */     for (String option : options) {
/*  32 */       this.options.add(option);
/*     */     }
/*     */   }
/*     */   
/*     */   public Set<String> getOptions() {
/*  37 */     return this.options;
/*     */   }
/*     */   
/*     */   public CommandLineOption hasArgument(Class<?> argumentType) {
/*  41 */     this.argumentType = argumentType;
/*  42 */     return this;
/*     */   }
/*     */   
/*     */   public CommandLineOption hasArgument() {
/*  46 */     this.argumentType = String.class;
/*  47 */     return this;
/*     */   }
/*     */   
/*     */   public CommandLineOption hasArguments() {
/*  51 */     this.argumentType = List.class;
/*  52 */     return this;
/*     */   }
/*     */   
/*     */   public String getDescription() {
/*  56 */     StringBuilder result = new StringBuilder();
/*  57 */     if (this.description != null) {
/*  58 */       result.append(this.description);
/*     */     }
/*  60 */     if (this.deprecationWarning != null) {
/*  61 */       if (result.length() > 0) {
/*  62 */         result.append(' ');
/*     */       }
/*  64 */       result.append("[deprecated - ");
/*  65 */       result.append(this.deprecationWarning);
/*  66 */       result.append("]");
/*     */     } 
/*  68 */     if (this.incubating) {
/*  69 */       if (result.length() > 0) {
/*  70 */         result.append(' ');
/*     */       }
/*  72 */       result.append("[incubating]");
/*     */     } 
/*  74 */     return result.toString();
/*     */   }
/*     */   
/*     */   public CommandLineOption hasDescription(String description) {
/*  78 */     this.description = description;
/*  79 */     return this;
/*     */   }
/*     */   
/*     */   public boolean getAllowsArguments() {
/*  83 */     return (this.argumentType != void.class);
/*     */   }
/*     */   
/*     */   public boolean getAllowsMultipleArguments() {
/*  87 */     return (this.argumentType == List.class);
/*     */   }
/*     */   
/*     */   public CommandLineOption deprecated(String deprecationWarning) {
/*  91 */     this.deprecationWarning = deprecationWarning;
/*  92 */     return this;
/*     */   }
/*     */   
/*     */   public CommandLineOption incubating() {
/*  96 */     this.incubating = true;
/*  97 */     return this;
/*     */   }
/*     */   
/*     */   public String getDeprecationWarning() {
/* 101 */     return this.deprecationWarning;
/*     */   }
/*     */   
/*     */   Set<CommandLineOption> getGroupWith() {
/* 105 */     return this.groupWith;
/*     */   }
/*     */   
/*     */   void groupWith(Set<CommandLineOption> options) {
/* 109 */     this.groupWith.addAll(options);
/* 110 */     this.groupWith.remove(this);
/*     */   }
/*     */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\CommandLineOption.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */