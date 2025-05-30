using System;
using System.Collections.Generic;

namespace ExamApp.Api.Models.Dtos
{
    public class UserProgramDto
    {
        public int Id { get; set; }
        public string UserId { get; set; } = string.Empty;
        public string ProgramName { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public DateTime CreatedDate { get; set; }
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public bool IsActive { get; set; }
        public string StudyType { get; set; } = string.Empty;
        public string StudyDuration { get; set; } = string.Empty;
        public int? QuestionsPerDay { get; set; }
        public int SubjectsPerDay { get; set; }
        public string RestDays { get; set; } = string.Empty;
        public string DifficultSubjects { get; set; } = string.Empty;
        public List<UserProgramScheduleDto> Schedules { get; set; } = new List<UserProgramScheduleDto>();
    }

    public class UserProgramScheduleDto
    {
        public int Id { get; set; }
        public int UserProgramId { get; set; }
        public DateTime ScheduleDate { get; set; }
        public int SubjectId { get; set; }
        public string SubjectName { get; set; } = string.Empty;
        public int? StudyDurationMinutes { get; set; }
        public int? QuestionCount { get; set; }
        public bool IsCompleted { get; set; }
        public DateTime? CompletedDate { get; set; }
        public string? Notes { get; set; }
    }
}
