/*    */ package org.gradle.cli;
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
/*    */ 
/*    */ public class SystemPropertiesCommandLineConverter
/*    */   extends AbstractPropertiesCommandLineConverter
/*    */ {
/*    */   protected String getPropertyOption() {
/* 22 */     return "D";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDetailed() {
/* 27 */     return "system-prop";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDescription() {
/* 32 */     return "Set system property of the JVM (e.g. -Dmyprop=myvalue).";
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\cli\SystemPropertiesCommandLineConverter.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */