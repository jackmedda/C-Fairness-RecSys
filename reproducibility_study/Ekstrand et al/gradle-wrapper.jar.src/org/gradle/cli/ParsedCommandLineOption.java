/*    */ package org.gradle.cli;
/*    */ 
/*    */ import java.util.ArrayList;
/*    */ import java.util.List;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ public class ParsedCommandLineOption
/*    */ {
/* 22 */   private final List<String> values = new ArrayList<String>();
/*    */   
/*    */   public String getValue() {
/* 25 */     if (!hasValue()) {
/* 26 */       throw new IllegalStateException("Option does not have any value.");
/*    */     }
/* 28 */     if (this.values.size() > 1) {
/* 29 */       throw new IllegalStateException("Option has multiple values.");
/*    */     }
/* 31 */     return this.values.get(0);
/*    */   }
/*    */   
/*    */   public List<String> getValues() {
/* 35 */     return this.values;
/*    */   }
/*    */   
/*    */   public void addArgument(String argument) {
/* 39 */     this.values.add(argument);
/*    */   }
/*    */   
/*    */   public boolean hasValue() {
/* 43 */     return !this.values.isEmpty();
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\ParsedCommandLineOption.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */