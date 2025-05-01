using ExamApp.Api.Data;
using ExamApp.Api.Helpers;
using ExamApp.Api.Models;
using ExamApp.Api.Models.Dtos;
using ExamApp.Api.Models.Dtos.Events;
using ExamApp.Api.Services.Interfaces;
using Microsoft.EntityFrameworkCore;
using System.Globalization;
using System.Numerics;
using System.Text.Json;

namespace ExamApp.Api.Services;

public class ExamService : IExamService
{
    private readonly AppDbContext _context;

    private readonly ImageHelper _imageHelper;
    private readonly IMinIoService _minioService;
    public ExamService(AppDbContext context, ImageHelper imageHelper, IMinIoService minioService)
    {
        _imageHelper = imageHelper;
        _context = context;
        _minioService = minioService;
    }

    public async Task<Paged<WorksheetDto>> GetWorksheetsForTeacherAsync(ExamFilterDto dto, User user, Teacher? teacher = null)
    {
        var query = _context.Worksheets.AsQueryable();

        if (dto.id > 0)
        {
            query = query.Where(t => t.Id == dto.id);
        }
        else
        {
            if (dto.gradeId.HasValue)
                query = query.Where(t => t.GradeId == dto.gradeId.Value);

            if (dto.subjectIds != null && dto.subjectIds.Any())
                query = query.Where(t => t.SubjectId.HasValue && dto.subjectIds.Contains(t.SubjectId.Value));

            if (dto.bookTestId > 0)
                query = query.Where(t => t.BookTestId == dto.bookTestId);

            if (!string.IsNullOrEmpty(dto.search))
            {
                var normalizedSearch = dto.search.ToLower(new CultureInfo("tr-TR"));
                query = query.Where(t =>
                    EF.Functions.Like(t.Name.ToLower(), $"%{normalizedSearch}%") ||
                    (t.Subtitle != null && EF.Functions.Like(t.Subtitle.ToLower(), $"%{normalizedSearch}%")) ||
                    EF.Functions.Like(t.Description.ToLower(), $"%{normalizedSearch}%"));
            }
        }
        // Her worksheet için kaç benzersiz öğrenci instance oluşturmuş

        var totalCount = await query.CountAsync(); // Toplam kayıt sayısı
        var tests = await query
            .Include(t => t.BookTest)
                .ThenInclude(bt => bt.Book)
            .Include(t => t.WorksheetQuestions)
                .ThenInclude(tq => tq.Question)
            .OrderBy(t => t.Name) // Sıralama için
            .Skip((dto.pageNumber - 1) * dto.pageSize) // Sayfalama için
            .Take(dto.pageSize)
            .ToListAsync();

        var worksheetDtos = tests.Select(t =>
        {
            return new WorksheetDto
            {
                Id = t.Id,
                Name = t.Name,
                Description = t.Description,
                GradeId = t.GradeId,
                SubjectId = t.SubjectId,
                MaxDurationSeconds = t.MaxDurationSeconds,
                IsPracticeTest = t.IsPracticeTest,
                Subtitle = t.Subtitle,
                ImageUrl = t.ImageUrl,
                BadgeText = t.BadgeText,
                BookTestId = t.BookTestId,
                BookId = t.BookTest?.BookId,
                QuestionCount = t.WorksheetQuestions.Count()
            };
        }).ToList();

        return new Paged<WorksheetDto>
        {
            PageNumber = dto.pageNumber,
            PageSize = dto.pageSize,
            TotalCount = totalCount,
            Items = worksheetDtos
        };
    }

    public async Task<Paged<WorksheetDto>> GetWorksheetsForStudentsAsync(ExamFilterDto dto, User user, Student? student = null)
    {
        var instanceQuery = _context.TestInstances
           .Where(ti => ti.StudentId == student.Id)
           .Include(ti => ti.WorksheetInstanceQuestions)
           .Include(ti => ti.Worksheet)
           .AsQueryable();


        var query = _context.Worksheets.AsQueryable();

        if (dto.id > 0)
        {
            query = query.Where(t => t.Id == dto.id);
            instanceQuery = instanceQuery.Where(ti => ti.WorksheetId == dto.id);
        }
        else
        {
            if (dto.gradeId.HasValue)
                query = query.Where(t => t.GradeId == dto.gradeId.Value);
            else if (student != null)
                query = query.Where(t => t.GradeId == student.GradeId);

            if (dto.subjectIds != null && dto.subjectIds.Any())
                query = query.Where(t => t.SubjectId.HasValue && dto.subjectIds.Contains(t.SubjectId.Value));

            if (dto.bookTestId > 0)
                query = query.Where(t => t.BookTestId == dto.bookTestId);

            if (!string.IsNullOrEmpty(dto.search))
            {
                var normalizedSearch = dto.search.ToLower(new CultureInfo("tr-TR"));
                query = query.Where(t =>
                    EF.Functions.Like(t.Name.ToLower(), $"%{normalizedSearch}%") ||
                    (t.Subtitle != null && EF.Functions.Like(t.Subtitle.ToLower(), $"%{normalizedSearch}%")) ||
                    EF.Functions.Like(t.Description.ToLower(), $"%{normalizedSearch}%"));
            }
        }
        // Her worksheet için kaç benzersiz öğrenci instance oluşturmuş
        var worksheetStudentCounts = await _context.TestInstances
            .GroupBy(ti => ti.WorksheetId)
            .Select(g => new
            {
                WorksheetId = g.Key,
                UniqueStudentCount = g.Select(ti => ti.StudentId).Distinct().Count()
            })
            .ToDictionaryAsync(x => x.WorksheetId, x => x.UniqueStudentCount);

        var instances = await instanceQuery.ToListAsync(); // 🔥 Burada `ToListAsync()` çağırarak veriyi hafızaya alıyoruz

        var totalCount = await query.CountAsync(); // Toplam kayıt sayısı
        var tests = await query
            .Include(t => t.BookTest)
                .ThenInclude(bt => bt.Book)
            .Include(t => t.WorksheetQuestions)
                .ThenInclude(tq => tq.Question)
            .OrderBy(t => t.Name) // Sıralama için
            .Skip((dto.pageNumber - 1) * dto.pageSize) // Sayfalama için
            .Take(dto.pageSize)
            .ToListAsync();

        var worksheetDtos = tests.Select(t =>
        {
            var instance = instances.FirstOrDefault(i => i.WorksheetId == t.Id);

            InstanceSummaryDto? instanceDto = null;
            if (instance != null)
            {
                var correct = instance.WorksheetInstanceQuestions.Count(wiq =>
                    wiq.SelectedAnswerId != null &&
                    t.WorksheetQuestions.Any(wq => wq.Id == wiq.WorksheetQuestionId
                        && wq.Question.CorrectAnswerId == wiq.SelectedAnswerId));

                var wrong = instance.WorksheetInstanceQuestions.Count(wiq =>
                    wiq.SelectedAnswerId != null &&
                    t.WorksheetQuestions.Any(wq => wq.Id == wiq.WorksheetQuestionId
                        && wq.Question.CorrectAnswerId != wiq.SelectedAnswerId));
                instanceDto = new InstanceSummaryDto
                {
                    Id = instance.Id,
                    Name = t.Name,
                    Status = (int)instance.Status,
                    ImageUrl = t.ImageUrl,
                    CompletedDate = instance.EndTime ?? DateTime.UtcNow,
                    DurationMinutes = instance.EndTime.HasValue ?
                (int)(instance.EndTime.Value - instance.StartTime).TotalMinutes : 0,
                    TotalQuestions = instance.WorksheetInstanceQuestions.Count,
                    CorrectAnswers = correct,
                    WrongAnswers = wrong,
                    Score = (correct * 100) / (instance.WorksheetInstanceQuestions.Count > 0 ? instance.WorksheetInstanceQuestions.Count : 1)
                };
            }

            return new WorksheetDto
            {
                Id = t.Id,
                Name = t.Name,
                Description = t.Description,
                GradeId = t.GradeId,
                SubjectId = t.SubjectId,
                MaxDurationSeconds = t.MaxDurationSeconds,
                IsPracticeTest = t.IsPracticeTest,
                Subtitle = t.Subtitle,
                ImageUrl = t.ImageUrl,
                BadgeText = t.BadgeText,
                BookTestId = t.BookTestId,
                BookId = t.BookTest?.BookId,
                QuestionCount = t.WorksheetQuestions.Count(),
                Instance = instanceDto, // 💡 Eklenen alan
                InstanceCount = worksheetStudentCounts.TryGetValue(t.Id, out var count) ? count : 0 // 💡 Yeni eklenen alan
            };
        }).ToList();

        return new Paged<WorksheetDto>
        {
            PageNumber = dto.pageNumber,
            PageSize = dto.pageSize,
            TotalCount = totalCount,
            Items = worksheetDtos
        };
    }

    public async Task<Paged<InstanceSummaryDto>> GetCompletedTestsAsync(Student student, int pageNumber, int pageSize)
    {
        var query = await _context.TestInstances
            .Where(wi => wi.StudentId == student.Id && wi.Status == WorksheetInstanceStatus.Completed)
            .Select(wi => new
            {
                wi.Id,
                wi.Worksheet.Name,
                wi.Worksheet.ImageUrl,
                wi.EndTime,
                wi.StartTime,
                TotalQuestions = wi.WorksheetInstanceQuestions.Count(),

                CorrectAnswers = wi.WorksheetInstanceQuestions.Count(wiq =>
                    wiq.SelectedAnswerId != null &&
                    wi.Worksheet.WorksheetQuestions.Any(wq =>
                        wq.Id == wiq.WorksheetQuestionId &&
                        wq.Question.CorrectAnswerId == wiq.SelectedAnswerId)),

                WrongAnswers = wi.WorksheetInstanceQuestions.Count(wiq =>
                    wiq.SelectedAnswerId != null &&
                    wi.Worksheet.WorksheetQuestions.Any(wq =>
                        wq.Id == wiq.WorksheetQuestionId &&
                        wq.Question.CorrectAnswerId != wiq.SelectedAnswerId))
            })
            .ToListAsync();

        var results = query.Select(wi => new InstanceSummaryDto
        {
            Id = wi.Id,
            Name = wi.Name,
            ImageUrl = wi.ImageUrl,
            CompletedDate = wi.EndTime ?? DateTime.UtcNow,
            DurationMinutes = wi.EndTime.HasValue ?
                (int)(wi.EndTime.Value - wi.StartTime).TotalMinutes : 0,
            TotalQuestions = wi.TotalQuestions,
            CorrectAnswers = wi.CorrectAnswers,
            WrongAnswers = wi.WrongAnswers,
            Score = (wi.CorrectAnswers * 100) / (wi.TotalQuestions > 0 ? wi.TotalQuestions : 1)
        })
        .OrderByDescending(wi => wi.CompletedDate)
        .Skip((pageNumber - 1) * pageSize)
        .Take(pageSize)
        .ToList();

        return new Paged<InstanceSummaryDto>
        {
            PageNumber = pageNumber,
            PageSize = pageSize,
            TotalCount = query.Count,
            Items = results
        };
    }

    public async Task<List<WorksheetDto>> GetLatestWorksheetsAsync(int pageNumber, int pageSize)
    {
        return await _context.Worksheets
            .OrderByDescending(t => t.CreateTime)
            .Skip((pageNumber - 1) * pageSize)
            .Take(pageSize)
            .Select(t => new WorksheetDto
            {
                Id = t.Id,
                Name = t.Name,
                Description = t.Description,
                GradeId = t.GradeId,
                SubjectId = t.SubjectId,
                MaxDurationSeconds = t.MaxDurationSeconds,
                IsPracticeTest = t.IsPracticeTest,
                Subtitle = t.Subtitle,
                ImageUrl = t.ImageUrl,
                BadgeText = t.BadgeText,
                BookTestId = t.BookTestId,
                BookId = t.BookTest != null ? t.BookTest.BookId : null,
                QuestionCount = t.WorksheetQuestions.Count()
            })
            .ToListAsync();
    }

    public async Task<List<QuestionDto>> GetExamQuestionsAsync()
    {
        return await _context.Questions
            .Include(q => q.Subject)
            .Select(q => new QuestionDto
            {
                Id = q.Id,
                Text = q.Text,
                SubText = q.SubText,
                ImageUrl = q.ImageUrl,
                CategoryName = q.Subject.Name,
                Point = q.Point
            })
            .ToListAsync();
    }

    public async Task<WorksheetDto?> GetWorksheetByIdAsync(int id)
    {
        var worksheet = await _context.Worksheets
            .Include(t => t.BookTest)
                .ThenInclude(bt => bt.Book)
            .Include(t => t.WorksheetQuestions)
                .ThenInclude(tq => tq.Question)
            .FirstOrDefaultAsync(t => t.Id == id);

        if (worksheet == null)
            return null;

        return new WorksheetDto
        {
            Id = worksheet.Id,
            Name = worksheet.Name,
            Description = worksheet.Description,
            GradeId = worksheet.GradeId,
            SubjectId = worksheet.SubjectId,
            MaxDurationSeconds = worksheet.MaxDurationSeconds,
            IsPracticeTest = worksheet.IsPracticeTest,
            Subtitle = worksheet.Subtitle,
            ImageUrl = worksheet.ImageUrl,
            BadgeText = worksheet.BadgeText,
            BookTestId = worksheet.BookTestId,
            BookId = worksheet.BookTest?.BookId,
            QuestionCount = worksheet.WorksheetQuestions.Count()
        };
    }

    public async Task<List<WorksheetWithInstanceDto>> GetWorksheetAndInstancesAsync(Student student, int gradeId)
    {
        var worksheets = await _context.Worksheets
            .Include(w => w.WorksheetQuestions)
            .Include(w => w.BookTest)
            .Where(w => w.GradeId == gradeId)
            .ToListAsync();

        var worksheetIds = worksheets.Select(w => w.Id).ToList();

        var instances = await _context.TestInstances
            .Where(i => worksheetIds.Contains(i.WorksheetId) && i.StudentId == student.Id)
            .ToListAsync();

        var result = worksheets.Select(w => new WorksheetWithInstanceDto
        {
            Worksheet = new WorksheetDto
            {
                Id = w.Id,
                Name = w.Name,
                Description = w.Description,
                GradeId = w.GradeId,
                SubjectId = w.SubjectId,
                MaxDurationSeconds = w.MaxDurationSeconds,
                IsPracticeTest = w.IsPracticeTest,
                Subtitle = w.Subtitle,
                ImageUrl = w.ImageUrl,
                BadgeText = w.BadgeText,
                BookTestId = w.BookTestId,
                BookId = w.BookTest?.BookId,
                QuestionCount = w.WorksheetQuestions.Count
            },
            Instance = instances.FirstOrDefault(i => i.WorksheetId == w.Id)
        }).ToList();

        return result;
    }

    public async Task<TestStartResultDto> StartTestAsync(int testId, Student student)
    {
        var existing = await _context.TestInstances
            .FirstOrDefaultAsync(ti => ti.StudentId == student.Id && ti.WorksheetId == testId
                && ti.EndTime == null
                && ti.Student.UserId == student.UserId);


        if (existing != null)
        {
            if (existing.Status == WorksheetInstanceStatus.Completed)
            {
                return new TestStartResultDto
                {
                    Success = false,
                    Message = "Bu test zaten tamamlanmış.",
                    InstanceId = existing.Id,
                    StartTime = existing.StartTime
                };
            }

            return new TestStartResultDto
            {
                Success = true,
                InstanceId = existing.Id,
                StartTime = existing.StartTime
            };
        }

        var instance = new WorksheetInstance
        {
            WorksheetId = testId,
            StudentId = student.Id,
            Status = WorksheetInstanceStatus.Started,
            WorksheetInstanceQuestions = new List<WorksheetInstanceQuestion>(),
            StartTime = DateTime.UtcNow
        };

        // Teste ait soruları TestQuestion tablosundan çekiyoruz
        var testQuestions = await _context.TestQuestions
            .Where(tq => tq.TestId == testId)
            .OrderBy(tq => tq.Order)
            .ToListAsync();

        foreach (var tq in testQuestions)
        {
            instance.WorksheetInstanceQuestions.Add(new WorksheetInstanceQuestion
            {
                WorksheetQuestionId = tq.Id,
                IsCorrect = false,
                TimeTaken = 0
            });
        }

        _context.TestInstances.Add(instance);
        await _context.SaveChangesAsync(); // burada audit çalışır

        return new TestStartResultDto
        {
            Success = true,
            InstanceId = instance.Id,
            StartTime = instance.StartTime
        };
    }

    public async Task<WorksheetInstanceDto?> GetTestInstanceQuestionsAsync(int testInstanceId, int userId)
    {
        var instance = await _context.TestInstances
            .Include(ti => ti.Worksheet)
            .Include(ti => ti.WorksheetInstanceQuestions)
                .ThenInclude(tiq => tiq.WorksheetQuestion)
                .ThenInclude(wq => wq.Question)
                    .ThenInclude(q => q.Answers)
            .Include(ti => ti.WorksheetInstanceQuestions)
                .ThenInclude(tiq => tiq.WorksheetQuestion)
                .ThenInclude(wq => wq.Question)
                    .ThenInclude(q => q.Passage)
            .FirstOrDefaultAsync(ti => ti.Id == testInstanceId && ti.Student.UserId == userId);

        if (instance == null)
            return null;

        return new WorksheetInstanceDto
        {
            Id = instance.Id,
            TestName = instance.Worksheet.Name,
            Status = instance.Status,
            MaxDurationSeconds = instance.Worksheet.MaxDurationSeconds,
            IsPracticeTest = instance.Worksheet.IsPracticeTest,
            TestInstanceQuestions = instance.WorksheetInstanceQuestions.Select(tiq => new WorksheetInstanceQuestionDto
            {
                Id = tiq.Id,
                Order = tiq.WorksheetQuestion.Order,
                SelectedAnswerId = tiq.SelectedAnswerId,
                Question = new QuestionDto
                {
                    Id = tiq.WorksheetQuestion.Question.Id,
                    Text = tiq.WorksheetQuestion?.Question?.Text ?? string.Empty,
                    SubText = tiq.WorksheetQuestion?.Question?.SubText,
                    ImageUrl = tiq.WorksheetQuestion?.Question?.ImageUrl,
                    IsExample = tiq.WorksheetQuestion?.Question?.IsExample ?? false,
                    PracticeCorrectAnswer = tiq.WorksheetQuestion?.Question?.PracticeCorrectAnswer,
                    AnswerColCount = tiq.WorksheetQuestion?.Question?.AnswerColCount ?? 0,
                    Passage = tiq.WorksheetQuestion?.Question?.PassageId != null
                        ? new PassageDto
                        {
                            Id = tiq.WorksheetQuestion.Question.Passage!.Id,
                            Title = tiq.WorksheetQuestion.Question.Passage.Title,
                            Text = tiq.WorksheetQuestion.Question.Passage.Text,
                            ImageUrl = tiq.WorksheetQuestion.Question.Passage.ImageUrl
                        }
                        : null,
                    Answers = tiq.WorksheetQuestion?.Question?.Answers?.Select(a => new AnswerDto
                    {
                        Id = a.Id,
                        Text = a.Text,
                        ImageUrl = a.ImageUrl
                    })?.ToList() ?? new List<AnswerDto>()
                }
            }).ToList()
        };
    }

    public async Task<List<WorksheetInstanceQuestionDto>> GetAllCanvasQuestions(bool includeAnswers = false, int maxId = 0)
    {
        var questions = await _context.Questions
            .Include(q => q.Answers)
            .Where(q => q.IsCanvasQuestion && q.Id > maxId)
            .Select(q => new WorksheetInstanceQuestionDto
            {
                Question = new QuestionDto
                {
                    Id = q.Id,
                    X = q.X,
                    Y = q.Y,
                    Width = q.Width,
                    Height = q.Height,
                    ImageUrl = q.ImageUrl,
                    Answers = includeAnswers ? q.Answers.Select(a => new AnswerDto
                    {
                        X = a.X,
                        Y = a.Y,
                        Width = a.Width,
                        Height = a.Height
                    }).ToList() : new List<AnswerDto>(),
                }
            }
            )
            .ToListAsync();

        return questions;
    }

    public async Task<WorksheetInstanceResultDto?> GetCanvasTestResultAsync(int testInstanceId, int userId, bool includeCorrectAnswer = false)
    {
        var testInstance = await _context.TestInstances
            .Include(ti => ti.Worksheet)
            .Include(ti => ti.WorksheetInstanceQuestions)
                .ThenInclude(tiq => tiq.WorksheetQuestion)
                .ThenInclude(tq => tq.Question)
                .ThenInclude(q => q.Answers)
            .Include(ti => ti.WorksheetInstanceQuestions)
                .ThenInclude(tiq => tiq.WorksheetQuestion)
                .ThenInclude(tq => tq.Question)
                .ThenInclude(q => q.Passage)
            .FirstOrDefaultAsync(ti => ti.Id == testInstanceId &&
                    ti.Student.UserId == userId);

        if (testInstance == null)
        {
            return null;
        }

        if (includeCorrectAnswer && testInstance.Status != WorksheetInstanceStatus.Completed)
        {
            return null;
        }

        var response = new WorksheetInstanceResultDto
        {
            Id = testInstance.Id,
            TestName = testInstance.Worksheet.Name,
            Status = testInstance.Status,
            MaxDurationSeconds = testInstance.Worksheet.MaxDurationSeconds,
            IsPracticeTest = testInstance.Worksheet.IsPracticeTest,
            TestInstanceQuestions = testInstance.WorksheetInstanceQuestions.Select(tiq => new WorksheetInstanceQuestionDto
            {
                Id = tiq.Id,
                Order = tiq.WorksheetQuestion.Order,
                Question = new QuestionDto
                {
                    Id = tiq.WorksheetQuestion.Question.Id,
                    Text = tiq.WorksheetQuestion?.Question?.Text ?? string.Empty,
                    SubText = tiq.WorksheetQuestion.Question.SubText,
                    ImageUrl = tiq.WorksheetQuestion.Question.ImageUrl,
                    IsExample = tiq.WorksheetQuestion.Question.IsExample,
                    CorrectAnswerId = includeCorrectAnswer ? tiq.WorksheetQuestion.Question.CorrectAnswerId : null,
                    Passage = tiq.WorksheetQuestion.Question.PassageId.HasValue ? new PassageDto
                    {
                        Id = tiq.WorksheetQuestion.Question.Passage?.Id,
                        Title = tiq.WorksheetQuestion.Question.Passage?.Title,
                        Text = tiq.WorksheetQuestion.Question.Passage?.Text,
                        ImageUrl = tiq.WorksheetQuestion.Question.Passage?.ImageUrl,
                        X = tiq.WorksheetQuestion.Question.Passage?.X,
                        Y = tiq.WorksheetQuestion.Question.Passage?.Y,
                        Width = tiq.WorksheetQuestion.Question.Passage?.Width,
                        Height = tiq.WorksheetQuestion.Question.Passage?.Height
                    } : null,
                    PracticeCorrectAnswer = tiq.WorksheetQuestion.Question.PracticeCorrectAnswer,
                    AnswerColCount = tiq.WorksheetQuestion.Question.AnswerColCount,
                    IsCanvasQuestion = tiq.WorksheetQuestion.Question.IsCanvasQuestion,
                    X = tiq.WorksheetQuestion.Question.X,
                    Y = tiq.WorksheetQuestion.Question.Y,
                    Width = tiq.WorksheetQuestion.Question.Width,
                    Height = tiq.WorksheetQuestion.Question.Height,
                    Answers = tiq.WorksheetQuestion.Question.Answers.Select(a => new AnswerDto
                    {
                        Id = a.Id,
                        Text = a.Text,
                        ImageUrl = a.ImageUrl,
                        X = a.X,
                        Y = a.Y,
                        Width = a.Width,
                        Height = a.Height
                    }).ToList()
                },
                SelectedAnswerId = tiq.SelectedAnswerId // Önceden seçilen cevap
            }).ToList()
        };


        return response;
    }

    public async Task<ResponseBaseDto> SaveAnswer(SaveAnswerDto dto, int userId)
    {
        var testInstanceQuestion = await _context.TestInstanceQuestions                    
                    .Include(t => t.WorksheetQuestion)
                    .ThenInclude(wq => wq.Question)
                    .ThenInclude(wiq => wiq.Subject)
            .FirstOrDefaultAsync(tiq => tiq.WorksheetInstanceId == dto.TestInstanceId &&
                tiq.Id == dto.TestQuestionId
                && tiq.WorksheetInstance.Student.UserId == userId);

        if (testInstanceQuestion == null)
        {
            return new ResponseBaseDto
            {
                Success = false,
                Message = "Test instance question not found."
            };
        }
        testInstanceQuestion.SelectedAnswerId = dto.SelectedAnswerId;
        testInstanceQuestion.TimeTaken = dto.TimeTaken;
        _context.TestInstanceQuestions.Update(testInstanceQuestion);

        // 1. Event oluştur
        var evt = new AnswerSubmittedEvent
        {
            UserId = userId,
            QuestionId = testInstanceQuestion.WorksheetQuestion.QuestionId,
            Subject = testInstanceQuestion.WorksheetQuestion.Question.Subject.Name,
            TestInstanceId = testInstanceQuestion.WorksheetInstanceId,
            SelectedAnswerId = dto.SelectedAnswerId,
            SubmittedAt = DateTime.UtcNow,
            TimeTakenInSeconds = dto.TimeTaken
        };

        // 2. Outbox'a yaz
        var outbox = new OutboxMessage
        {
            Type = nameof(AnswerSubmittedEvent),
            Content = JsonSerializer.Serialize(evt)
        };
        _context.OutboxMessages.Add(outbox);


        // Update Question Count
        await _context.SaveChangesAsync();
        return new ResponseBaseDto
        {
            Success = true,
            Message = "Answer saved successfully."
        };
    }

    public async Task<ResponseBaseDto> EndTest(int testInstanceId, int userId)
    {
        var testInstance = await _context.TestInstances
            .FirstOrDefaultAsync(ti => ti.Id == testInstanceId && ti.Student.UserId == userId);

        if (testInstance == null)
        {
            return new ResponseBaseDto
            {
                Success = false,
                Message = "Test instance not found."
            };
        }

        if (testInstance.Status != WorksheetInstanceStatus.Started)
        {
            return new ResponseBaseDto
            {
                Success = false,
                Message = $"Bu test zaten {testInstance.Status} durumunda."
            };
        }

        testInstance.Status = WorksheetInstanceStatus.Completed;
        testInstance.EndTime = DateTime.UtcNow;
        await _context.SaveChangesAsync();

        return new ResponseBaseDto
        {
            Success = true,
            Message = "Test ended successfully."
        };
    }

    public async Task<ExamSavedDto> CreateOrUpdateAsync(ExamDto examDto, int userId)
    {
        if (examDto == null)
        {
            // TODO: bu kontrolleri contrrollerda yapabilirsin
            // return BadRequest(new { error = "Sınav bilgileri eksik!" });
            return new ExamSavedDto
            {
                Success = false,
                Message = "Sınav bilgileri eksik!"
            };
        }

        try
        {
            if (examDto.BookId == 0 && string.IsNullOrWhiteSpace(examDto.NewBookName))
            {
                // return BadRequest(new { error = "Kitap seçilmedi!" });
                return new ExamSavedDto
                {
                    Success = false,
                    Message = "Kitap seçilmedi!"
                };
            }

            if (examDto.BookTestId == 0 && string.IsNullOrWhiteSpace(examDto.NewBookTestName))
            {
                // return BadRequest(new { error = "Kipta Test seçilmedi!" });
                return new ExamSavedDto
                {
                    Success = false,
                    Message = "Kipta Test seçilmedi!"
                };
            }

            Book? book = null;

            if (examDto.BookId == 0)
            {
                if (string.IsNullOrWhiteSpace(examDto.NewBookName))
                {
                    // return BadRequest(new { error = "Kitap seçilmedi!" });
                    return new ExamSavedDto
                    {
                        Success = false,
                        Message = "Kitap seçilmedi!"
                    };
                    // return BadRequest(new { error = "Kitap seçilmedi!" });
                }

                if (string.IsNullOrWhiteSpace(examDto.NewBookTestName))
                {
                    // return BadRequest(new { error = "Kipta Test seçilmedi!" });
                    return new ExamSavedDto
                    {
                        Success = false,
                        Message = "Kipta Test seçilmedi!"
                    };
                }

                book = new Book
                {
                    Name = examDto.NewBookName
                };

                book.BookTests =
                [
                  new BookTest
                  {
                      Name = examDto.NewBookTestName,
                      BookId = book.Id
                  },
                ];

                _context.Books.Add(book);
                await _context.SaveChangesAsync();
            }
            else
            {
                book = await _context.Books
                            .Include(b => b.BookTests)
                        .FirstOrDefaultAsync(b => b.Id == examDto.BookId);

                if (book == null)
                {
                    return new ExamSavedDto
                    {
                        Success = false,
                        Message = "Kitap bulunamadı!"
                    };
                    // return NotFound(new { error = "Kitap bulunamadı!" });
                }
                if (examDto.BookTestId == 0)
                {
                    if (string.IsNullOrWhiteSpace(examDto.NewBookTestName))
                    {
                        // return BadRequest(new { error = "Kipta Test seçilmedi!" });
                        return new ExamSavedDto
                        {
                            Success = false,
                            Message = "Kipta Test seçilmedi!"
                        };
                    }
                    else
                    {
                        book.BookTests.Add(new BookTest
                        {
                            Name = examDto.NewBookTestName,
                            BookId = book.Id
                        });
                    }
                    await _context.SaveChangesAsync();
                }
            }


            var bookTestId = examDto.BookTestId == 0 ?
                        book.BookTests.First(bt => bt.Name == examDto.NewBookTestName).Id : examDto.BookTestId;

            Worksheet? examination;
            // 📌 Eğer ID varsa, veritabanından o soruyu bulup güncelle
            if (examDto.Id > 0)
            {
                examination = await _context.Worksheets.FindAsync(examDto.Id);

                if (examination == null)
                {
                    // return NotFound(new { error = "Test bulunamadı!" });
                    return new ExamSavedDto
                    {
                        Success = false,
                        Message = "Test bulunamadı!"
                    };
                }

                examination.Name = examDto.Name;
                examination.Description = examDto.Description;
                examination.GradeId = examDto.GradeId;
                examination.MaxDurationSeconds = examDto.MaxDurationSeconds;
                examination.IsPracticeTest = examDto.IsPracticeTest;
                examination.Subtitle = examDto.Subtitle;
                examination.BookTestId = bookTestId;

                // 📌 Eğer yeni resim varsa, güncelle
                if (!string.IsNullOrEmpty(examDto.ImageUrl) &&
                    _imageHelper.IsBase64String(examDto.ImageUrl))
                {
                    byte[] imageBytes = Convert.FromBase64String(examDto.ImageUrl.Split(',')[1]);
                    await using var imageStream = new MemoryStream(imageBytes);
                    examination.ImageUrl = await _minioService.UploadFileAsync(imageStream, $"{Guid.NewGuid()}.jpg", "exams");
                }

                _context.Worksheets.Update(examination);
            }
            else
            {
                // 📌 Yeni Soru Oluştur (INSERT)
                examination = new Worksheet
                {
                    Name = examDto.Name,
                    Description = examDto.Description,
                    GradeId = examDto.GradeId,
                    MaxDurationSeconds = examDto.MaxDurationSeconds,
                    IsPracticeTest = examDto.IsPracticeTest,
                    Subtitle = examDto.Subtitle,
                    BookTestId = bookTestId
                };

                // 📌 Eğer yeni resim varsa, güncelle
                if (!string.IsNullOrEmpty(examDto.ImageUrl) &&
                    _imageHelper.IsBase64String(examDto.ImageUrl))
                {
                    byte[] imageBytes = Convert.FromBase64String(examDto.ImageUrl.Split(',')[1]);
                    await using var imageStream = new MemoryStream(imageBytes);
                    examination.ImageUrl = await _minioService.UploadFileAsync(imageStream, $"{Guid.NewGuid()}.jpg", "exams");
                }

                _context.Worksheets.Add(examination);
            }

            // 📌 Değişiklikleri Kaydet
            await _context.SaveChangesAsync();

            return new ExamSavedDto
            {
                Message = examDto.Id > 0 ?
                            "Test başarıyla güncellendi!" : "Test başarıyla kaydedildi!",
                ExamId = examination.Id,
                BookId = book?.Id,
                BookTestId = bookTestId
            };
        }
        catch (Exception ex)
        {
            return new ExamSavedDto
            {
                Success = false,
                Message = ex.Message
            };
            // return BadRequest(new { error = ex.Message });
        }
    }

    public async Task<ExamAllStatisticsDto> GetGroupedStudentStatistics(int studentId)
    {
        var testInstances = await _context.TestInstances
            .Where(ti => ti.StudentId == studentId)
            .Include(ti => ti.Worksheet)
            .Include(ti => ti.WorksheetInstanceQuestions)
                .ThenInclude(wiq => wiq.WorksheetQuestion)
                    .ThenInclude(wq => wq.Question)
            .ToListAsync();

        // 🔹 Total verileri için
        int totalSolved = testInstances.Count;
        var completedTests = testInstances.Where(ti => ti.Status == WorksheetInstanceStatus.Completed).ToList();
        int completedCount = completedTests.Count;
        int totalCorrect = 0;
        int totalWrong = 0;
        int totalTimeTaken = 0;

        foreach (var instance in completedTests)
        {
            foreach (var question in instance.WorksheetInstanceQuestions)
            {
                totalTimeTaken += question.TimeTaken / 60;

                if (question.SelectedAnswerId.HasValue)
                {
                    var correctAnswerId = question.WorksheetQuestion.Question.CorrectAnswerId;
                    if (correctAnswerId.HasValue)
                    {
                        if (question.SelectedAnswerId == correctAnswerId)
                            totalCorrect++;
                        else
                            totalWrong++;
                    }
                }
            }
        }

        // 🔹 Gruplama verileri
        var grouped = testInstances
            .GroupBy(ti => new { ti.Worksheet.GradeId, ti.Worksheet.Name })
            .Select(group =>
            {
                var allTests = group.ToList();
                var completed = allTests.Where(ti => ti.Status == WorksheetInstanceStatus.Completed).ToList();

                int groupCorrect = 0;
                int groupWrong = 0;
                int groupTimeTaken = 0;

                foreach (var instance in completed)
                {
                    foreach (var question in instance.WorksheetInstanceQuestions)
                    {
                        groupTimeTaken += question.TimeTaken / 60;

                        if (question.SelectedAnswerId.HasValue)
                        {
                            var correctAnswerId = question.WorksheetQuestion.Question.CorrectAnswerId;
                            if (correctAnswerId.HasValue)
                            {
                                if (question.SelectedAnswerId == correctAnswerId)
                                    groupCorrect++;
                                else
                                    groupWrong++;
                            }
                        }
                    }
                }

                return new
                {
                    GradeId = group.Key.GradeId,
                    TestName = group.Key.Name,
                    TotalSolvedTests = allTests.Count,
                    CompletedTests = completed.Count,
                    TotalTimeSpentMinutes = groupTimeTaken,
                    TotalCorrectAnswers = groupCorrect,
                    TotalWrongAnswers = groupWrong
                };
            })
            .ToList();

        // 🔹 Response
        var response = new ExamAllStatisticsDto
        {
            Total = new ExamStatisticsDto
            {
                TotalSolvedTests = totalSolved,
                CompletedTests = completedCount,
                TotalTimeSpentMinutes = totalTimeTaken,
                TotalCorrectAnswers = totalCorrect,
                TotalWrongAnswers = totalWrong
            },
            Grouped = grouped.Select(g => new ExamStatisticsDto
            {
                TotalSolvedTests = g.TotalSolvedTests,
                CompletedTests = g.CompletedTests,
                TotalTimeSpentMinutes = g.TotalTimeSpentMinutes,
                TotalCorrectAnswers = g.TotalCorrectAnswers,
                TotalWrongAnswers = g.TotalWrongAnswers
            }).ToList()
        };

        return response;
    }

}
