using System;
using System.ComponentModel.DataAnnotations;

namespace ExamApp.Api.Data;

public class WorksheetQuestion
{
    [Key]
    public int Id { get; set; }
    public int TestId { get; set; }
    public Worksheet Worksheet { get; set; }
    public int Order { get; set; } // 🟢 Test içinde soru sırası
    public int QuestionId { get; set; }
    public Question Question { get; set; }    

}

